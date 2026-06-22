"""Корректность распределённого обучения.

Ключевой инвариант DDP: усреднение градиентов через all-reduce по W рангам,
где каждый ранг считает средний градиент по своей доле батча, ЭКВИВАЛЕНТНО
одному процессу, обработавшему весь глобальный батч (среднее средних при равных
долях = общее среднее). measure_ddp_vs_single() это проверяет численно — на CPU
(gloo), без GPU. Это и есть «доказал, а не запустил».
"""
from __future__ import annotations

import json
import os
import socket
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from .model import GPT, GPTConfig

# Маленькая модель без weight tying — чтобы тест был быстрым и численно чистым.
_TINY = GPTConfig(
    vocab_size=32, block_size=16, n_layer=2, n_head=2, n_embd=32, tie_weights=False
)


def _build_model(seed: int = 0) -> GPT:
    torch.manual_seed(seed)
    return GPT(_TINY)


def _global_batch(world_size: int, per_rank_bs: int, seed: int = 1234):
    gen = torch.Generator().manual_seed(seed)
    total = world_size * per_rank_bs
    x = torch.randint(0, _TINY.vocab_size, (total, _TINY.block_size), generator=gen)
    y = torch.randint(0, _TINY.vocab_size, (total, _TINY.block_size), generator=gen)
    return x, y


def _grad_vector(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _worker(rank: int, world_size: int, per_rank_bs: int, port: int, result_path: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ.setdefault("USE_LIBUV", "0")  # Windows-сборки torch без libuv
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    model = _build_model(seed=0)  # одинаковая инициализация на всех рангах
    ddp = DDP(model)

    x, y = _global_batch(world_size, per_rank_bs)
    xs = x.chunk(world_size)[rank]
    ys = y.chunk(world_size)[rank]

    ddp.zero_grad()
    _, loss = ddp(xs, ys)
    loss.backward()  # DDP усредняет градиенты через all-reduce

    if rank == 0:
        ref = _build_model(seed=0)  # эталон: один процесс, весь глобальный батч
        ref.zero_grad()
        _, ref_loss = ref(x, y)
        ref_loss.backward()
        diff = (_grad_vector(model) - _grad_vector(ref)).abs().max().item()
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({"max_grad_diff": diff}, f)

    dist.destroy_process_group()


def measure_ddp_vs_single(world_size: int = 2, per_rank_bs: int = 4) -> float:
    """Максимальное расхождение градиента DDP (world_size процессов, gloo/CPU)
    и эквивалентного одиночного процесса с полным батчем. ~0 => DDP корректен.
    """
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    port = _free_port()
    try:
        mp.spawn(_worker, args=(world_size, per_rank_bs, port, path), nprocs=world_size, join=True)
        with open(path, encoding="utf-8") as f:
            return float(json.load(f)["max_grad_diff"])
    finally:
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # Windows-консоль: кириллица в stdout
    diff = measure_ddp_vs_single(world_size=2, per_rank_bs=4)
    print(f"max |grad_DDP - grad_single| = {diff:.2e}")
    print("OK: DDP эквивалентен одиночному процессу" if diff < 1e-4 else "FAIL")
