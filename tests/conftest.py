"""Делает пакет minigpt импортируемым — в т.ч. в spawn-процессах DDP-теста
(дочерние процессы наследуют PYTHONPATH из окружения).
"""
import os
import sys
from pathlib import Path

SRC = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, SRC)
os.environ["PYTHONPATH"] = SRC + os.pathsep + os.environ.get("PYTHONPATH", "")
