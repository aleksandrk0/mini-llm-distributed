"""Рисует эффективность масштабирования: compute-bound vs comm-bound, в один график.

Читает bench/scaling_compute.csv (обязательно) и bench/scaling_comm.csv (если есть),
строит эффективность = tok/с(N) / (N · tok/с(1)) против идеальной линии 1.0.
Запуск:  python bench/plot.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _load(name: str):
    path = ROOT / "bench" / name
    if not path.exists():
        return None
    with path.open(encoding="utf-8-sig") as f:  # utf-8-sig снимает BOM, если есть
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["world_size"]))
    ws = [int(r["world_size"]) for r in rows]
    tps = [float(r["tokens_per_s"]) for r in rows]
    eff = [tps[i] / (ws[i] * tps[0]) for i in range(len(ws))]
    return ws, eff


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    compute = _load("scaling_compute.csv")
    comm = _load("scaling_comm.csv")
    if compute is None:
        raise SystemExit("нет bench/scaling_compute.csv — сначала прогони run_l2.sh")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ws = compute[0]
    ax.plot(ws, [1.0] * len(ws), "--", color="gray", label="идеальное линейное")
    ax.plot(compute[0], compute[1], "o-", color="#2563eb", linewidth=2,
            label="compute-bound (200M, b32/s512)")
    for x, y in zip(compute[0], compute[1], strict=False):
        ax.annotate(f"{y * 100:.0f}%", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center")
    if comm is not None:
        ax.plot(comm[0], comm[1], "s-", color="#dc2626", linewidth=2,
                label="comm-bound (85M, b16/s256)")
        for x, y in zip(comm[0], comm[1], strict=False):
            ax.annotate(f"{y * 100:.0f}%", (x, y), textcoords="offset points",
                        xytext=(0, -15), ha="center")

    ax.set_xlabel("число GPU")
    ax.set_ylabel("эффективность масштабирования")
    ax.set_xticks(ws)
    ax.set_ylim(0, 1.12)
    ax.set_title("Масштабирование: compute-bound vs communication-bound")
    ax.legend()
    fig.tight_layout()
    out = ROOT / "bench" / "scaling.png"
    fig.savefig(out, dpi=120)
    print(f"сохранено: {out}")


if __name__ == "__main__":
    main()
