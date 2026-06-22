"""Базовая одиночная тренировка (без распределёнки) — точка отсчёта.

Запуск:  python train_single.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.stdout.reconfigure(encoding="utf-8")

import torch  # noqa: E402

from minigpt import GPT, GPTConfig  # noqa: E402
from minigpt.data import get_batch, get_dataset  # noqa: E402


def main(steps: int = 200, batch_size: int = 32, block_size: int = 64, lr: float = 3e-4) -> None:
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data, vocab = get_dataset()
    model = GPT(GPTConfig(vocab_size=vocab, block_size=block_size)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(0)

    model.train()
    t0 = time.perf_counter()
    tokens = 0
    loss = torch.tensor(0.0)
    for step in range(steps):
        x, y = get_batch(data, block_size, batch_size, generator=gen, device=device)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        tokens += x.numel()
        if step % 50 == 0:
            print(f"step {step:4d}  loss {loss.item():.3f}")

    dt = time.perf_counter() - t0
    print(f"итог: loss {loss.item():.3f}  {tokens / dt:,.0f} ток/с  "
          f"({device}, params={model.num_params():,})")


if __name__ == "__main__":
    main()
