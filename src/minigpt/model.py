"""Компактный GPT (decoder-only трансформер), nano-масштаб.

Цель — не SOTA, а корректная и читаемая база для демонстрации распределённого
обучения. Causal-внимание через scaled_dot_product_attention (флеш-ядро в torch>=2),
weight tying эмбеддингов и выходной головы.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    tie_weights: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        head = c // self.n_head
        q = q.view(b, t, self.n_head, head).transpose(1, 2)
        k = k.view(b, t, self.n_head, head).transpose(1, 2)
        v = v.view(b, t, self.n_head, head).transpose(1, 2)
        drop = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(self.act(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.tok.weight = self.head.weight  # weight tying (до init: общий тензор)

        # GPT-2 инициализация: normal(0, 0.02); residual-проекции масштабируются на
        # 1/sqrt(2*n_layer) — иначе дисперсия растёт с глубиной (важно на L16+).
        self.apply(self._init_weights)
        std = 0.02 / math.sqrt(2 * cfg.n_layer)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):  # attn.proj и mlp.proj — выход residual
                nn.init.normal_(p, mean=0.0, std=std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        _, t = idx.shape
        assert t <= self.cfg.block_size, f"seq {t} > block_size {self.cfg.block_size}"
        pos = torch.arange(t, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
