"""
交互式对话：自动包装 `<s>[INST] ... [/INST] `，自回归生成；支持流式打印。
默认加载 chat_model.pth（SFT 后）；可用 --checkpoint 指定。
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F

from config import Config, merge_config
from dataset import CharTokenizer
from model import CharGPT
def load_chat_checkpoint(path: str, device: torch.device) -> tuple[CharGPT, CharTokenizer, Config]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    config = merge_config(ckpt["config"])
    tokenizer = CharTokenizer.from_ordered_chars(ckpt["chars"])
    model = CharGPT(config, vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, tokenizer, config


def stream_reply(
    model: CharGPT,
    tokenizer: CharTokenizer,
    user_text: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
) -> None:
    """
    仅发送前缀 `<s>[INST] {user} [/INST] `，模型续写 answer 与 `</s>`。
    流式：每步打印一个字符并 flush。
    """
    prefix = f"<s>[INST] {user_text} [/INST] "
    ids = tokenizer.encode_checked(prefix)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    stop_suffix = "</s>"
    acc = ""

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -model.config.block_size :]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
            ch = tokenizer.decode([next_id.item()])
            print(ch, end="", flush=True)
            acc += ch
            if acc.endswith(stop_suffix):
                break
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Char-GPT SFT 交互对话（流式输出）")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="chat_model.pth",
        help="SFT 保存的权重，默认 chat_model.pth",
    )
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    model, tokenizer, _ = load_chat_checkpoint(args.checkpoint, device)

    print("输入内容后回车生成；空行或 Ctrl+D / Ctrl+Z 结束。")
    print("与训练一致：自动加前缀 `<s>[INST] 你的输入 [/INST] `，模型续写直至 `</s>`。\n")

    try:
        while True:
            try:
                line = input("用户> ").strip()
            except EOFError:
                break
            if not line:
                break
            stream_reply(
                model,
                tokenizer,
                line,
                device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
    except KeyboardInterrupt:
        print("\n退出。", file=sys.stderr)


if __name__ == "__main__":
    main()
