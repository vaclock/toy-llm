"""
从本地纯文本构建字符词表，并产生 (x, y) 下一字符预测对。
x[t] 预测目标为 y[t] = x[t+1]，最后一个位置无标签则丢弃或由实现约定。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator, Tuple

import torch
from torch.utils.data import Dataset


class CharTokenizer:
    """char <-> idx 双向映射。"""

    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def encode_checked(self, s: str) -> list[int]:
        """逐字检查是否在词表中；SFT 等场景避免静默错误。"""
        out: list[int] = []
        for i, c in enumerate(s):
            if c not in self.stoi:
                raise KeyError(
                    f"字符 {c!r}（位置 {i}）不在词表中；请把该字符加入预训练语料 input.txt 后重训 base 模型。"
                )
            out.append(self.stoi[c])
        return out

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    @classmethod
    def from_ordered_chars(cls, chars: list[str]) -> "CharTokenizer":
        """与训练时相同的词表顺序：索引 i 对应字符 chars[i]。"""
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        obj = cls.__new__(cls)
        obj.stoi = stoi
        obj.itos = itos
        obj.vocab_size = len(chars)
        return obj


class CharTextDataset(Dataset):
    """
    将全文编码为长序列，按 block_size 切块；
    每块 x 长度为 block_size，y 为右移一位的下一个 token。
    """

    def __init__(self, text: str, tokenizer: CharTokenizer, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        # 有效块数：需要 x 与 y 各长 block_size，故总长至少 block_size+1
        self.num_chunks = (len(self.data) - 1) // block_size

    def __len__(self) -> int:
        return self.num_chunks

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.block_size
        chunk = self.data[start : start + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def load_text_and_tokenizer(path: str | Path) -> Tuple[str, CharTokenizer]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    return text, tokenizer


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从长序列 data 中随机采样 batch_size 个起始位置，构造 (x, y)。
    data: 一维 LongTensor，长度 >= block_size + 1
    """
    max_start = len(data) - block_size - 1
    ix = torch.randint(max_start + 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


def iter_train_batches(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """无限随机 batch 迭代器（用于 train 循环）。"""
    while True:
        yield get_batch(data, block_size, batch_size, device)
