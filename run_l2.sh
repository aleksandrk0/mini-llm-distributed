#!/usr/bin/env bash
# Turnkey L2: воспроизводит ВСЕ числа README одной командой. Конфиги закреплены
# здесь явно (не через внешний env) — `bash run_l2.sh` даёт ровно те цифры, что
# в README. Артефакты: bench/scaling_compute.csv, scaling_comm.csv, optim.csv, scaling.png.
#   bash run_l2.sh
set -e

N=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "Обнаружено GPU: ${N}"
[ "${N}" -lt 1 ] && { echo "GPU не найдены"; exit 1; }
pip install -q matplotlib >/dev/null 2>&1 || true
rm -f bench/scaling_compute.csv bench/scaling_comm.csv

# --- Свип 1: communication-bound (мелкая модель 85M, GPU недогружены) ---
for w in 1 2 4; do
  if [ "${w}" -le "${N}" ]; then
    echo "=== comm-bound: ${w} GPU ==="
    CSV_OUT=bench/scaling_comm.csv \
      N_LAYER=12 N_HEAD=12 N_EMBD=768 BATCH=16 BLOCK=256 \
      torchrun --nproc_per_node="${w}" bench/scaling.py
  fi
done

# --- Свип 2: compute-bound (модель 200M, GPU насыщены) ---
for w in 1 2 4; do
  if [ "${w}" -le "${N}" ]; then
    echo "=== compute-bound: ${w} GPU ==="
    CSV_OUT=bench/scaling_compute.csv \
      N_LAYER=16 N_HEAD=16 N_EMBD=1024 BATCH=32 BLOCK=512 \
      torchrun --nproc_per_node="${w}" bench/scaling.py
  fi
done

# --- Память DDP vs FSDP (та же модель 200M) ---
echo "=== память DDP vs FSDP (${N} GPU, модель 200M) ==="
CFG="N_LAYER=16 N_HEAD=16 N_EMBD=1024 BATCH=32 BLOCK=512"
env ${CFG} torchrun --nproc_per_node="${N}" train_ddp.py  | tail -1
env ${CFG} torchrun --nproc_per_node="${N}" train_fsdp.py | tail -1

# --- Лестница ускорений (1 GPU) ---
echo "=== ускорения (1 GPU) ==="
python bench/optim.py

python bench/plot.py
echo ""
echo "ГОТОВО. Закоммить: bench/scaling_compute.csv, scaling_comm.csv, optim.csv, scaling.png"
