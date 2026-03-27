"""
从 base_model.pth 加载权重，对 JSONL 做 SFT，保存 chat_model.pth。
"""

from __future__ import annotations

import dataclasses

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import Config, merge_config
from dataset import CharTokenizer
from model import CharGPT
from sft_dataset import SFTJsonlDataset, assert_sft_template_in_vocab


def load_base_checkpoint(path: str, device: torch.device) -> tuple[CharGPT, CharTokenizer, Config]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    config = merge_config(ckpt["config"])
    tokenizer = CharTokenizer.from_ordered_chars(ckpt["chars"])
    model = CharGPT(config, vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, tokenizer, config


def infinite_batches(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def main() -> None:
    config = Config()
    device = torch.device(
        config.device if torch.cuda.is_available() else "cpu"
    )
    print(f"device: {device}")

    base_path = config.base_checkpoint_path
    model, tokenizer, ckpt_config = load_base_checkpoint(base_path, device)
    assert_sft_template_in_vocab(tokenizer)
    # 训练时用当前 Config 中的 SFT 与 ignore_index，其余结构以 checkpoint 为准
    ckpt_config.sft_learning_rate = config.sft_learning_rate
    ckpt_config.sft_max_iters = config.sft_max_iters
    ckpt_config.sft_log_interval = config.sft_log_interval
    ckpt_config.sft_weight_decay = config.sft_weight_decay
    ckpt_config.ignore_index = config.ignore_index
    train_config = ckpt_config

    dataset = SFTJsonlDataset(config.sft_data_path, tokenizer, train_config)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    it = infinite_batches(loader)

    optimizer = AdamW(
        model.parameters(),
        lr=config.sft_learning_rate,
        weight_decay=config.sft_weight_decay,
    )

    model.train()
    for step in range(1, config.sft_max_iters + 1):
        x, y = next(it)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % config.sft_log_interval == 0 or step == 1:
            print(f"sft step {step:6d} | loss {loss.item():.4f}")

    chars_ordered = [tokenizer.itos[i] for i in range(tokenizer.vocab_size)]
    out_path = config.chat_checkpoint_path
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": dataclasses.asdict(train_config),
            "chars": chars_ordered,
        },
        out_path,
    )
    print(f"saved SFT checkpoint -> {out_path}")


if __name__ == "__main__":
    main()
