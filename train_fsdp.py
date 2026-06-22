"""FSDP-тренировка (Fully Sharded Data Parallel ≈ ZeRO-3) под torchrun. Нужен GPU.

Шардирует параметры, градиенты и состояние оптимизатора по рангам → кратно
меньше памяти на GPU, чем DDP. Демонстрирует: auto-wrap по блокам трансформера,
mixed precision (bf16), activation checkpointing, FULL_SHARD.

  torchrun --nproc_per_node=4 train_fsdp.py
"""
from __future__ import annotations

import functools
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (  # noqa: E402
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (  # noqa: E402
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp import (  # noqa: E402
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy  # noqa: E402

from minigpt import GPT, GPTConfig  # noqa: E402
from minigpt.data import get_batch, get_dataset  # noqa: E402
from minigpt.model import Block  # noqa: E402


def main(steps: int = 200, per_rank_bs: int = 32, block_size: int = 128, lr: float = 3e-4) -> None:
    rank = int(os.environ["RANK"])
    local = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ["WORLD_SIZE"])
    if not torch.cuda.is_available():
        raise SystemExit("FSDP в этом примере рассчитан на GPU/nccl (см. RUNBOOK)")

    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}")
    is_main = rank == 0
    if is_main:
        sys.stdout.reconfigure(encoding="utf-8")

    torch.manual_seed(0)
    data, vocab = get_dataset()
    model = GPT(GPTConfig(vocab_size=vocab, block_size=block_size, tie_weights=False)).to(device)

    wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
    mixed = MixedPrecision(
        param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16
    )
    fsdp = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mixed,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=local,
    )
    apply_activation_checkpointing(
        fsdp,
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ),
        check_fn=lambda m: isinstance(m, Block),
    )

    opt = torch.optim.AdamW(fsdp.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(1000 + rank)
    fsdp.train()
    t0 = time.perf_counter()
    tokens = 0
    for step in range(steps):
        x, y = get_batch(data, block_size, per_rank_bs, generator=gen, device=device)
        _, loss = fsdp(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        tokens += x.numel() * world
        if is_main and step % 50 == 0:
            print(f"step {step:4d}  loss {loss.item():.3f}")

    if is_main:
        dt = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"итог FSDP (world={world}): {tokens / dt:,.0f} ток/с, peak GPU mem {peak:.2f} GB")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
