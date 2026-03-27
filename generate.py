"""
加载 base_model.pth，自回归生成文本。
"""

from __future__ import annotations

import argparse

import torch

from config import Config, merge_config
from dataset import CharTokenizer
from model import CharGPT


def load_model_from_checkpoint(path: str, device: torch.device) -> tuple[CharGPT, CharTokenizer, Config]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # 旧版 PyTorch 无 weights_only 参数
        ckpt = torch.load(path, map_location=device)
    config = merge_config(ckpt["config"])
    tokenizer = CharTokenizer.from_ordered_chars(ckpt["chars"])
    model = CharGPT(config, vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, tokenizer, config


def main() -> None:
    parser = argparse.ArgumentParser(description="Char-GPT 自回归生成")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="base_model.pth",
        help="训练保存的权重路径",
    )
    parser.add_argument("--prompt", type=str, default="", help="起始提示（可为空）")
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    model, tokenizer, config = load_model_from_checkpoint(args.checkpoint, device)

    if args.prompt:
        ids = tokenizer.encode(args.prompt)
    else:
        ids = [0]

    idx = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=args.max_new_tokens)

    generated = out[0].tolist()
    text = tokenizer.decode(generated)
    print(text, end="")


if __name__ == "__main__":
    main()
