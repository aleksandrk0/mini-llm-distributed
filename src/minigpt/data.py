"""Игрушечный char-level датасет (встроенный текст, без скачиваний).

Поток токенов: батч — это случайные окна длины block_size. Для распределённого
обучения каждый ранг берёт окна со своим сидом (непересекающиеся выборки) —
аналог DistributedSampler для map-датасетов.
"""
from __future__ import annotations

import torch

_TEXT = (
    "distributed training splits work across many accelerators. "
    "data parallel keeps a full model copy on each device and averages gradients. "
    "fully sharded data parallel shards parameters gradients and optimizer state. "
    "tensor parallel splits matrices inside a layer across devices. "
    "pipeline parallel places consecutive layers on different devices and streams microbatches. "
    "the all reduce collective sums gradients across ranks. "
    "communication cost and memory budget decide which strategy to use. "
) * 8


def get_dataset() -> tuple[torch.Tensor, int]:
    chars = sorted(set(_TEXT))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in _TEXT], dtype=torch.long)
    return data, len(chars)


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    generator: torch.Generator | None = None,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    high = len(data) - block_size - 1
    ix = torch.randint(high, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)
