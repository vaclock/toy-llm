"""
JSONL 指令微调数据：模板 `<s>[INST] {msg} [/INST] {ans} </s>`，
仅对 answer 与结束符 `</s>` 计算 CrossEntropy（指令位置 target = ignore_index）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from config import Config
from dataset import CharTokenizer


def format_sft_sequence(msg: str, ans: str) -> str:
    """与常见 INST 模板一致；字符级，无额外 BPE。"""
    return f"<s>[INST] {msg} [/INST] {ans} </s>"


def assert_sft_template_in_vocab(tokenizer: CharTokenizer) -> None:
    """确保预训练词表含 SFT 模板字符；否则提示扩充 input.txt 并重训 base。"""
    try:
        # 仅用空串占位，避免用 "a"/"b" 等 ASCII 字母（纯中文语料词表常不含）
        tokenizer.encode_checked(format_sft_sequence("", ""))
    except KeyError as e:
        raise RuntimeError(
            "SFT 模板字符不在词表中。请在 input.txt 中加入模板所需字符（例如单独一行 "
            "`<s>[INST] [/INST] </s>`），重新运行 `python train.py` 生成新的 base_model.pth。"
        ) from e


def _pad_token_id(tokenizer: CharTokenizer) -> int:
    if " " in tokenizer.stoi:
        return tokenizer.stoi[" "]
    return 0


def build_sft_example(
    tokenizer: CharTokenizer,
    msg: str,
    ans: str,
    block_size: int,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    返回 x, y，形状均为 [block_size]。
    y 在指令与 padding 位置为 ignore_index；仅 answer（含 `</s>`）参与 loss。
    若截断后不含完整 answer 起始，返回 None。
    """
    full = format_sft_sequence(msg, ans)
    ans_start = full.find(ans)
    if ans_start < 0:
        return None

    ids = tokenizer.encode_checked(full)
    max_tokens = block_size + 1  # x、y 各 block_size，共需 block_size+1 个 token
    if len(ids) > max_tokens:
        drop = len(ids) - max_tokens
        ids = ids[-max_tokens:]
        ans_start -= drop
    if ans_start < 0 or ans_start >= len(ids):
        return None

    pad_id = _pad_token_id(tokenizer)
    real_len = len(ids)
    if real_len < max_tokens:
        ids = ids + [pad_id] * (max_tokens - real_len)

    # ids 长度 max_tokens = block_size + 1
    x = torch.tensor(ids[:-1], dtype=torch.long)
    y = torch.full((block_size,), ignore_index, dtype=torch.long)
    for t in range(block_size):
        pos = t + 1  # 预测 ids[pos]
        if pos >= real_len:
            y[t] = ignore_index
        elif pos >= ans_start:
            y[t] = ids[pos]
        else:
            y[t] = ignore_index

    return x, y


class SFTJsonlDataset(Dataset):
    """
    每行 JSON：`{"instruction": "...", "answer": "..."}`。
    仅保留能完整编码且截断后仍含 answer 起始的样本。
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: CharTokenizer,
        config: Config,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.ignore_index = config.ignore_index
        path = Path(path)
        raw: list[tuple[str, str]] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj: dict[str, Any] = json.loads(line)
                msg = obj["instruction"].strip()
                ans = obj["answer"].strip()
                if not msg or not ans:
                    continue
                raw.append((msg, ans))

        self.samples: list[tuple[str, str]] = []
        for msg, ans in raw:
            ex = build_sft_example(
                tokenizer,
                msg,
                ans,
                config.block_size,
                self.ignore_index,
            )
            if ex is not None:
                self.samples.append((msg, ans))

        if not self.samples:
            raise RuntimeError(
                f"SFT 无有效样本：请检查 {path} 是否与词表兼容，或增大 config.block_size。"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        msg, ans = self.samples[idx]
        out = build_sft_example(
            self.tokenizer,
            msg,
            ans,
            self.config.block_size,
            self.ignore_index,
        )
        assert out is not None
        return out
