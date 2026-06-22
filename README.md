# mini-llm-distributed

![CI](https://github.com/aleksandrk0/mini-llm-distributed/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Компактный GPT + **честная обвязка распределённого обучения** (DDP / FSDP) с тем,
что отличает senior от «запустил туториал»: **доказанной корректностью**, измеримым
масштабированием и честной заметкой о железе.

> Акцент проекта — не SOTA-модель, а **правильная механика распределёнки** и умение
> её проверить. Корректность DDP доказывается численно на CPU, без GPU.

---

## ⭐ Что доказано (проверяемо, без GPU)

| Проверка | Результат |
| --- | --- |
| **Корректность DDP** = одиночный процесс | `max\|grad_DDP − grad_single\| = 1.3e-08` |
| Тесты (модель, данные, DDP) | 6 passed |
| Throughput одиночного процесса (CPU, 0.8M параметров) | ~73 000 ток/с |

**Почему это важно.** DDP усредняет градиенты через all-reduce: каждый ранг считает
средний градиент по своей доле батча, и среднее средних (при равных долях) = общее
среднее по глобальному батчу. `tests/test_distributed.py` это **доказывает численно** —
DDP на 2 процессах даёт тот же градиент, что один процесс на полном батче.

```bash
python -m minigpt.distributed     # max |grad_DDP - grad_single| = 1.3e-08
```

---

## Таксономия параллелизма

| Стратегия | Что шардирует | Коммуникация | Когда применять |
| --- | --- | --- | --- |
| **DP / DDP** | ничего (полная копия модели на ранг) | all-reduce градиентов | модель влезает в 1 GPU, нужен throughput |
| **FSDP / ZeRO-3** | параметры + градиенты + состояние оптимизатора | all-gather + reduce-scatter | модель НЕ влезает по памяти |
| **Tensor parallel (TP)** | матрицы внутри слоя | all-reduce в каждом слое | огромные слои, быстрый интерконнект (NVLink) |
| **Pipeline parallel (PP)** | последовательные слои по устройствам | point-to-point между стадиями | очень глубокие модели; цена — «пузырь» конвейера |
| **Sequence parallel (SP)** | активации по длине последовательности | дополняет TP | длинный контекст |

### Когда что выбирать (коротко)
- Упёрся в **скорость**, память ок → **DDP**.
- Упёрся в **память** → **FSDP/ZeRO** (или TP, если интерконнект быстрый).
- Модель **глубокая**, не делится по памяти иначе → добавить **PP**.
- Огромные слои + NVLink → **TP**.
- Реальные большие прогоны комбинируют их — **3D-параллелизм** (DP × TP × PP),
  поверх FSDP/ZeRO для памяти.

---

## Запуск

```bash
pip install -r requirements.txt

python train_single.py                          # базовая тренировка (1 процесс)
python -m minigpt.distributed                   # доказательство корректности DDP
torchrun --nproc_per_node=2 train_ddp.py        # DDP (на CPU — gloo, для логики)
torchrun --nproc_per_node=4 train_fsdp.py       # FSDP (нужен GPU/nccl)
torchrun --nproc_per_node=4 bench/scaling.py    # точка кривой масштабирования
pytest                                          # тесты, включая корректность DDP
```

---

## Масштабирование (заполняется реальным multi-GPU прогоном)

`bench/scaling.py` под `torchrun --nproc_per_node=1|2|4` копит строки в
`bench/scaling.csv`. Эффективность масштабирования = `(tok/с при N) / (N × tok/с при 1)`.

| GPU | tokens/s | эффективность | peak mem/GPU |
| --- | --- | --- | --- |
| 1 | — | 1.00 | — |
| 2 | — | — | — |
| 4 | — | — | — |

> ⚠️ **Честно о железе.** Разработка и проверка корректности — на одной RTX 4090 и на
> CPU (gloo, мультипроцесс). Числа масштабирования заполняются **реальным прогоном на
> арендованных 2–4 GPU** — пошаговый [RUNBOOK_multigpu.md](RUNBOOK_multigpu.md). Здесь
> **нет выдуманных цифр**: пустые ячейки честнее правдоподобной выдумки.

---

## Подводные камни (учтены)
- **PP «пузырь»** — простой устройств в начале/конце конвейера; лечится микробатчами.
- **all-reduce overhead** — на медленном интерконнекте DDP упирается в связь, не в счёт.
- **Grad accumulation × DDP** — синхронизировать только на последнем микрошаге
  (`no_sync()`), иначе лишний all-reduce.
- **Tied weights × DDP** — общий параметр может ломать reducer; в тесте корректности
  weight tying отключён намеренно (`tie_weights=False`).

---

## Структура
```
mini-llm-distributed/
├── src/minigpt/
│   ├── model.py          # компактный GPT (causal attention, weight tying)
│   ├── data.py           # char-level датасет (встроенный, без скачиваний)
│   └── distributed.py    # ⭐ measure_ddp_vs_single — доказательство корректности
├── train_single.py       # базовая тренировка
├── train_ddp.py          # DDP под torchrun (nccl/gloo)
├── train_fsdp.py         # FSDP: auto-wrap, bf16, activation checkpointing, FULL_SHARD
├── bench/scaling.py      # кривая throughput vs число GPU
├── tests/                # модель, данные, корректность DDP (gloo/CPU)
└── RUNBOOK_multigpu.md   # как получить реальные scaling-цифры на аренде
```

## Лицензия
MIT
