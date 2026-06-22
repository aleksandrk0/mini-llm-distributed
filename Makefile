.PHONY: install single ddp fsdp correctness scaling test lint

install:
	pip install -r requirements.txt

single:
	python train_single.py

correctness:
	python -m minigpt.distributed

ddp:
	torchrun --nproc_per_node=2 train_ddp.py

fsdp:
	torchrun --nproc_per_node=4 train_fsdp.py

scaling:
	torchrun --nproc_per_node=2 bench/scaling.py

test:
	pytest

lint:
	ruff check .
