# RUNBOOK: реальные scaling-цифры на арендованных GPU

Цель — снять настоящую кривую масштабирования за ~$1 и ~20–40 мин на 2–4 GPU (сам
счёт — секунды, время уходит на провижининг). Подходит vast.ai / runpod / любой
инстанс с несколькими GPU на одной ноде.

## 1. Инстанс
- **2× или 4× GPU на одной ноде** (например 2–4× RTX 3090/4090/A10).
- Образ с PyTorch ≥ 2.1 + CUDA. Для 50-й серии (Blackwell) нужен PyTorch ≥ 2.7 / CUDA 12.8.

## 2. Подготовка
```bash
git clone https://github.com/aleksandrk0/mini-llm-distributed
cd mini-llm-distributed
pip install -e .    # ставит пакет minigpt + зависимости
nvidia-smi          # убедиться, что видно N GPU
```
Возможные готчи (встречались на практике):
- Ubuntu 24.04 блокирует системный pip → `export PIP_BREAK_SYSTEM_PACKAGES=1` перед install.
- В образе нет PyTorch (`torchrun: command not found`) → `pip install torch`
  (под драйвер из `nvidia-smi`: например cu128 для CUDA 12.8).
- Нет `python` (только `python3`) → `ln -sf $(command -v python3) /usr/local/bin/python`.

## 3. Снять всё одной командой
```bash
bash run_l2.sh
```
Скрипт сам определит число GPU и с **закреплёнными в нём конфигами** снимет: два свипа
масштабирования (comm-bound 85M и compute-bound 200M), память DDP vs FSDP и лестницу
ускорений; затем построит график. Артефакты:
`bench/scaling_compute.csv`, `bench/scaling_comm.csv`, `bench/optim.csv`, `bench/scaling.png`.

OOM-демо (отдельно; `n_embd` обязан делиться на `n_head`):
```bash
STEPS=20 N_LAYER=32 N_HEAD=16 N_EMBD=2048 torchrun --nproc_per_node=4 train_ddp.py   # DDP -> CUDA OOM
STEPS=20 N_LAYER=32 N_HEAD=16 N_EMBD=2048 torchrun --nproc_per_node=4 train_fsdp.py  # FSDP влезает (~8 ГБ)
```

## 4. Закоммитить артефакты
Числа самодокументированы: в CSV есть колонки `params`/`config`, эффективность и график
строит `bench/plot.py` — **руками в README ничего пересчитывать и вписывать не надо**.
```bash
git add bench/scaling_compute.csv bench/scaling_comm.csv bench/optim.csv bench/scaling.png
git commit -m "bench: scaling/memory/optim на <твоё-железо>"
```
Если твои числа заметно отличаются от текущих в README — обнови таблицы под свой прогон
(под каждой указаны модель/precision/batch/seq).

## 5. Погасить инстанс
Остановить/удалить сразу после прогона, чтобы не капала аренда.

---
**Что показывает результат:** способность не просто запустить распределёнку, а
**измерить** её (throughput, эффективность, память) и подать воспроизводимо — ровно то,
что проверяют на секциях pretrain/distributed.
