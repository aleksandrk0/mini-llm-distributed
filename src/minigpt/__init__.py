"""mini-llm-distributed: компактный GPT + честная обвязка распределённого обучения
(DDP / FSDP) с проверкой корректности и бенчмарком масштабирования.
"""
from .model import GPT, GPTConfig

__all__ = ["GPT", "GPTConfig"]
