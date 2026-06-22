import pytest
import torch.distributed as dist

from minigpt.distributed import measure_ddp_vs_single


@pytest.mark.skipif(not dist.is_gloo_available(), reason="нужен gloo backend")
def test_ddp_grad_matches_single_process():
    # DDP на 2 процессах (gloo/CPU) должен дать тот же градиент, что и один
    # процесс на полном батче — доказательство корректности all-reduce усреднения.
    diff = measure_ddp_vs_single(world_size=2, per_rank_bs=4)
    assert diff < 1e-4, f"DDP отклоняется от одиночного процесса на {diff}"
