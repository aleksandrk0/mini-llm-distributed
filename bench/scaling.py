"""Бенчмарк масштабирования throughput. Запускать под torchrun на 1/2/4 GPU —
строки (world_size, tokens/s, peak_gb) копятся в bench/scaling.csv для кривой
масштабирования и эффективности (см. README).

  torchrun --nproc_per_node=1 bench/scaling.py
  torchrun --nproc_per_node=2 bench/scaling.py
  torchrun --nproc_per_node=4 bench/scaling.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402

from minigpt import GPT, GPTConfig  # noqa: E402
from minigpt.data import get_batch, get_dataset  # noqa: E402


def _envi(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def main(warmup: int = 5) -> None:
    # Конфиг compute-bound по умолчанию (GPU реально загружены -> честная кривая).
    # Переопределяется env: N_LAYER/N_HEAD/N_EMBD/BATCH/BLOCK/STEPS (для OOM-тюнинга).
    steps = _envi("STEPS", 100)
    per_rank_bs = _envi("BATCH", 16)
    block_size = _envi("BLOCK", 256)
    rank = int(os.environ["RANK"])
    local = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ["WORLD_SIZE"])
    cuda = torch.cuda.is_available()
    dist.init_process_group("nccl" if cuda else "gloo", rank=rank, world_size=world)
    if cuda:
        torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}" if cuda else "cpu")

    torch.manual_seed(0)
    data, vocab = get_dataset()
    cfg = GPTConfig(
        vocab_size=vocab,
        block_size=block_size,
        n_layer=_envi("N_LAYER", 12),
        n_head=_envi("N_HEAD", 12),
        n_embd=_envi("N_EMBD", 768),
    )
    model = GPT(cfg).to(device)
    ddp = DDP(model, device_ids=[local] if cuda else None)
    opt = torch.optim.AdamW(ddp.parameters(), lr=3e-4)
    gen = torch.Generator().manual_seed(1000 + rank)

    def step_once() -> None:
        x, y = get_batch(data, block_size, per_rank_bs, generator=gen, device=device)
        _, loss = ddp(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    for _ in range(warmup):
        step_once()
    if cuda:
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    tokens = 0
    for _ in range(steps):
        step_once()
        tokens += per_rank_bs * block_size * world
    if cuda:
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    if rank == 0:
        sys.stdout.reconfigure(encoding="utf-8")
        tps = tokens / dt
        peak = torch.cuda.max_memory_allocated() / 1e9 if cuda else 0.0
        nparams = model.num_params()
        conf = f"L{cfg.n_layer}_E{cfg.n_embd}_b{per_rank_bs}_s{block_size}_fp32"
        out = Path(os.environ.get("CSV_OUT", str(ROOT / "bench" / "scaling.csv")))
        header = not out.exists()
        with open(out, "a", encoding="utf-8") as f:
            if header:
                f.write("world_size,tokens_per_s,peak_gb,params,config\n")
            f.write(f"{world},{tps:.0f},{peak:.2f},{nparams},{conf}\n")
        print(f"world={world}  params={nparams:,}  {conf}  {tps:,.0f} ток/с  peak {peak:.2f} GB")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
