"""DDP-тренировка (Distributed Data Parallel) под torchrun.

Каждый ранг держит полную копию модели; градиенты усредняются all-reduce.
  1 GPU:   torchrun --nproc_per_node=1 train_ddp.py
  4 GPU:   torchrun --nproc_per_node=4 train_ddp.py
  CPU:     torchrun --nproc_per_node=2 train_ddp.py   (gloo, для проверки логики)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402

from minigpt import GPT, GPTConfig  # noqa: E402
from minigpt.data import get_batch, get_dataset  # noqa: E402


def _envi(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def main(lr: float = 3e-4) -> None:
    # Размер модели/батч из env (N_LAYER/N_HEAD/N_EMBD/BATCH/BLOCK/STEPS) — для
    # тяжёлых прогонов и OOM-демо без правки кода.
    steps = _envi("STEPS", 200)
    per_rank_bs = _envi("BATCH", 32)
    block_size = _envi("BLOCK", 256)
    rank = int(os.environ["RANK"])
    local = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ["WORLD_SIZE"])
    cuda = torch.cuda.is_available()
    dist.init_process_group("nccl" if cuda else "gloo", rank=rank, world_size=world)
    if cuda:
        torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}" if cuda else "cpu")
    is_main = rank == 0
    if is_main:
        sys.stdout.reconfigure(encoding="utf-8")

    torch.manual_seed(0)  # одинаковая инициализация на всех рангах
    data, vocab = get_dataset()
    cfg = GPTConfig(
        vocab_size=vocab, block_size=block_size,
        n_layer=_envi("N_LAYER", 12), n_head=_envi("N_HEAD", 12), n_embd=_envi("N_EMBD", 768),
    )
    model = GPT(cfg).to(device)
    ddp = DDP(model, device_ids=[local] if cuda else None)
    opt = torch.optim.AdamW(ddp.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(1000 + rank)  # непересекающиеся доли данных

    ddp.train()
    t0 = time.perf_counter()
    tokens = 0
    for step in range(steps):
        x, y = get_batch(data, block_size, per_rank_bs, generator=gen, device=device)
        _, loss = ddp(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        tokens += x.numel() * world
        if is_main and step % 50 == 0:
            print(f"step {step:4d}  loss {loss.item():.3f}")

    if is_main:
        tps = tokens / (time.perf_counter() - t0)
        msg = f"итог world={world}: {tps:,.0f} ток/с, params={model.num_params():,}"
        if cuda:
            msg += f", peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
        print(msg)
        torch.save(model.state_dict(), "ckpt_ddp.pt")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
