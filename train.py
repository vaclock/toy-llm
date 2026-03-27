"""
训练 Char-GPT：随机 batch，周期性打印 loss，保存 base_model.pth。
"""

from __future__ import annotations

import dataclasses

import torch
from torch.optim import AdamW

from config import Config
from dataset import iter_train_batches, load_text_and_tokenizer
from model import CharGPT


def main() -> None:
    config = Config()
    device = torch.device(
        config.device if torch.cuda.is_available() else "cpu"
    )
    print(f"device: {device}")

    text, tokenizer = load_text_and_tokenizer(config.data_path)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    model = CharGPT(config, vocab_size=tokenizer.vocab_size).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    batch_iter = iter_train_batches(
        data, config.block_size, config.batch_size, device
    )

    for step in range(1, config.max_iters + 1):
        x, y = next(batch_iter)
        logits, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % config.log_interval == 0 or step == 1:
            print(f"step {step:6d} | loss {loss.item():.4f}")

    # 保存权重与词表，供 generate 复现
    chars_ordered = [tokenizer.itos[i] for i in range(tokenizer.vocab_size)]
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": dataclasses.asdict(config),
        "chars": chars_ordered,
    }
    torch.save(checkpoint, config.checkpoint_path)
    print(f"saved checkpoint -> {config.checkpoint_path}")


if __name__ == "__main__":
    main()
