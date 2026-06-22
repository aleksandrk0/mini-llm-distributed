"""Лестница оптимизаций throughput на одной GPU: каждый приём добавляется поверх
предыдущего, видно его вклад в скорость.

  fp32 baseline -> +TF32 -> +bf16 (AMP) -> +torch.compile

Запуск:  python bench/optim.py     (нужен GPU; на A10/Ampere TF32 и bf16 дают много)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch  # noqa: E402

from minigpt import GPT, GPTConfig  # noqa: E402
from minigpt.data import get_batch, get_dataset  # noqa: E402

BLOCK = 256


def _throughput(model, data, steps, batch, device, amp_dtype=None) -> float:
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    gen = torch.Generator().manual_seed(0)

    def one_step() -> None:
        x, y = get_batch(data, BLOCK, batch, generator=gen, device=device)
        if amp_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    for _ in range(10):  # warmup (здесь же отрабатывает torch.compile)
        one_step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        one_step()
    torch.cuda.synchronize()
    return steps * batch * BLOCK / (time.perf_counter() - t0)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if not torch.cuda.is_available():
        raise SystemExit("нужен GPU")
    device = "cuda"
    data, vocab = get_dataset()
    batch = int(os.environ.get("BATCH", 16))
    steps = int(os.environ.get("STEPS", 50))

    def fresh():
        torch.manual_seed(0)
        cfg = GPTConfig(vocab_size=vocab, block_size=BLOCK, n_layer=12, n_head=12, n_embd=768)
        return GPT(cfg).to(device)

    results: list[tuple[str, float]] = []

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results.append(("fp32 baseline", _throughput(fresh(), data, steps, batch, device)))

    torch.set_float32_matmul_precision("high")  # включает TF32 на матмулах Ampere
    results.append(("+TF32", _throughput(fresh(), data, steps, batch, device)))

    results.append(("+bf16 AMP",
                    _throughput(fresh(), data, steps, batch, device, amp_dtype=torch.bfloat16)))

    try:
        compiled = torch.compile(fresh())
        tps = _throughput(compiled, data, steps, batch, device, amp_dtype=torch.bfloat16)
        results.append(("+torch.compile", tps))
    except Exception as e:  # noqa: BLE001
        print(f"torch.compile пропущен: {e}")

    base = results[0][1]
    print("\nОптимизация        tokens/s    ускорение")
    print("-" * 44)
    for name, tps in results:
        print(f"{name:<18} {tps:>9,.0f}  {tps / base:>6.2f}x")

    out = Path(__file__).parent / "optim.csv"
    with open(out, "w", encoding="utf-8") as f:
        f.write("optimization,tokens_per_s,speedup\n")
        for name, tps in results:
            f.write(f"{name},{tps:.0f},{tps / base:.2f}\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
