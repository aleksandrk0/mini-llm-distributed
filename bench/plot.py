"""Рисует кривую масштабирования из bench/scaling.csv -> bench/scaling.png.

Фактический throughput vs идеальное линейное ускорение; над точками —
эффективность масштабирования. Запуск:  python bench/plot.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    csv_path = ROOT / "bench" / "scaling.csv"
    if not csv_path.exists():
        raise SystemExit("Нет bench/scaling.csv — сначала прогони bench/scaling.py под torchrun")

    with csv_path.open(encoding="utf-8-sig") as f:  # utf-8-sig снимает BOM, если есть
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["world_size"]))
    ws = [int(r["world_size"]) for r in rows]
    tps = [float(r["tokens_per_s"]) for r in rows]
    base = tps[0]
    ideal = [base * w for w in ws]
    eff = [tps[i] / (ws[i] * base) for i in range(len(ws))]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ws, ideal, "--", color="gray", label="идеальное линейное")
    ax.plot(ws, tps, "o-", color="#2563eb", linewidth=2, label="фактический throughput")
    for i in range(len(ws)):
        ax.annotate(f"{eff[i] * 100:.0f}%", (ws[i], tps[i]),
                    textcoords="offset points", xytext=(0, 9), ha="center")
    ax.set_xlabel("число GPU")
    ax.set_ylabel("tokens / s")
    ax.set_xticks(ws)
    ax.set_title("Масштабирование DDP (% — эффективность)")
    ax.legend()
    fig.tight_layout()

    out = ROOT / "bench" / "scaling.png"
    fig.savefig(out, dpi=120)
    print(f"сохранено: {out}")


if __name__ == "__main__":
    main()
