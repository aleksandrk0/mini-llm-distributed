#!/usr/bin/env bash
# Turnkey L2: снять реальные scaling-цифры на multi-GPU инстансе одной командой.
#   bash run_l2.sh
# Затем закоммитить bench/scaling.csv + bench/scaling.png и обновить README.
set -e

N=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "Обнаружено GPU: ${N}"
[ "${N}" -lt 1 ] && { echo "GPU не найдены"; exit 1; }

pip install -q matplotlib >/dev/null 2>&1 || true
rm -f bench/scaling.csv

# Кривая масштабирования: 1, 2, 4 GPU (сколько есть)
for w in 1 2 4; do
  if [ "${w}" -le "${N}" ]; then
    echo "=== scaling: ${w} GPU ==="
    torchrun --nproc_per_node="${w}" bench/scaling.py
  fi
done

# Память: DDP vs FSDP при одинаковом конфиге
echo "=== память DDP vs FSDP (${N} GPU) ==="
torchrun --nproc_per_node="${N}" train_ddp.py  | tail -1
torchrun --nproc_per_node="${N}" train_fsdp.py | tail -1

python bench/plot.py
echo ""
echo "ГОТОВО. Артефакты:"
echo "  bench/scaling.csv  — точки (world_size, tokens/s, peak_gb)"
echo "  bench/scaling.png  — график для README"
echo "Скопируй их к себе, закоммить и впиши числа в раздел README «Масштабирование»."
