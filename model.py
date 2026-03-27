"""
字符级 GPT：手动实现 Multi-Head Attention、因果掩码、FFN(GELU)、残差与 LayerNorm。
未使用 nn.Transformer / nn.MultiheadAttention。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class CausalSelfAttention(nn.Module):
    """缩放点积注意力 + 下三角因果掩码；Q/K/V 为独立线性投影。"""

    def __init__(self, config: Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 注册下三角掩码，上三角为 True（将被填为 -inf）
        # 形状 [1, 1, T_max, T_max]，forward 时按当前 T 切片
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        B, T, C = x.shape

        q = self.q_proj(x)  # [B, T, C]
        k = self.k_proj(x)  # [B, T, C]
        v = self.v_proj(x)  # [B, T, C]

        # [B, T, C] -> [B, T, nh, hs] -> [B, nh, T, hs]
        nh = self.n_head
        hs = self.head_dim
        q = q.view(B, T, nh, hs).transpose(1, 2)  # [B, nh, T, hs]
        k = k.view(B, T, nh, hs).transpose(1, 2)  # [B, nh, T, hs]
        v = v.view(B, T, nh, hs).transpose(1, 2)  # [B, nh, T, hs]

        # 注意力分数: Q K^T / sqrt(d_k)
        # [B, nh, T, hs] @ [B, nh, hs, T] -> [B, nh, T, T]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))

        # 因果掩码：未来位置为 -inf，softmax 后为 0
        mask = self.causal_mask[:, :, :T, :T]
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)  # [B, nh, T, T]
        att = self.attn_dropout(att)

        # [B, nh, T, T] @ [B, nh, T, hs] -> [B, nh, T, hs]
        y = att @ v

        # [B, nh, T, hs] -> [B, T, nh, hs] -> [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        y = self.resid_dropout(y)
        return y


class FeedForward(nn.Module):
    """FFN: Linear -> GELU -> Linear（按 4 * n_embd 扩展，与常见 GPT 一致）。"""

    def __init__(self, config: Config):
        super().__init__()
        hidden = 4 * config.n_embd
        self.fc1 = nn.Linear(config.n_embd, hidden)
        self.fc2 = nn.Linear(hidden, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        x = self.fc1(x)  # [B, T, 4C]
        x = F.gelu(x)
        x = self.fc2(x)  # [B, T, C]
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Pre-LayerNorm + 残差 + 自注意力 + FFN。"""

    def __init__(self, config: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        x = x + self.attn(self.ln1(x))  # 残差 + 注意力
        x = x + self.ffn(self.ln2(x))  # 残差 + FFN
        return x


class CharGPT(nn.Module):
    """Token Embedding + 可学习位置 Embedding + n_layer 个 TransformerBlock + LM Head。"""

    def __init__(self, config: Config, vocab_size: int):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        # idx: [B, T], targets: [B, T] 或 None
        B, T = idx.shape
        assert T <= self.config.block_size

        tok = self.tok_emb(idx)  # [B, T, C]
        pos = self.pos_emb[:, :T, :]  # [1, T, C] 广播到 [B, T, C]
        x = self.drop(tok + pos)

        for block in self.blocks:
            x = block(x)  # [B, T, C]

        x = self.ln_f(x)  # [B, T, C]
        logits = self.lm_head(x)  # [B, T, vocab_size]

        loss = None
        if targets is not None:
            # 展平为 [B*T, V] 与 [B*T]；ignore_index 用于 SFT 仅对 answer 计 loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.config.ignore_index,
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """idx: [B, T]，自回归扩展序列。"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # [B, V]
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # [B, 1]
            idx = torch.cat((idx, next_id), dim=1)
        return idx
