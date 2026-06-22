# RUNBOOK: реальные scaling-цифры на арендованных GPU

Цель — получить **настоящую** кривую масштабирования (а не выдуманную) за ~5–8 $ и ~1 час
на 2–4 GPU. Подходит vast.ai / runpod / любой инстанс с несколькими GPU.

## 1. Инстанс
- Возьми инстанс с **2× или 4× GPU** (например 2–4× RTX 3090/4090/A10) на одной ноде.
- Образ с CUDA + PyTorch ≥ 2.1 (готовые pytorch-образы есть у обоих провайдеров).

## 2. Подготовка
```bash
git clone https://github.com/aleksandrk0/mini-llm-distributed
cd mini-llm-distributed
pip install -r requirements.txt
nvidia-smi          # убедиться, что видно N GPU
```

## 3. Снять точки кривой
```bash
torchrun --nproc_per_node=1 bench/scaling.py
torchrun --nproc_per_node=2 bench/scaling.py
torchrun --nproc_per_node=4 bench/scaling.py    # если 4 GPU
cat bench/scaling.csv
```
`bench/scaling.csv` соберёт строки `world_size,tokens_per_s,peak_gb`.

## 4. DDP vs FSDP по памяти (тот же конфиг)
```bash
torchrun --nproc_per_node=4 train_ddp.py        # запомнить peak mem/GPU
torchrun --nproc_per_node=4 train_fsdp.py       # peak mem/GPU должен быть ниже
```
FSDP шардирует параметры/градиенты/оптимизатор → меньше памяти на GPU, чем DDP.

## 5. Заполнить README
1. Перенести числа из `bench/scaling.csv` в таблицу «Масштабирование».
2. Посчитать эффективность: `(tok/с при N) / (N × tok/с при 1)`.
3. Вписать peak mem DDP vs FSDP.
4. Закоммитить `bench/scaling.csv` + обновлённый README.

## 6. Погасить инстанс
Остановить/удалить инстанс сразу после прогона, чтобы не капала аренда.

---
**Что показывает результат:** способность не просто запустить распределёнку, а
**измерить** её (throughput, эффективность, память) и честно отчитаться — ровно то,
что проверяют на секциях pretrain/distributed.
