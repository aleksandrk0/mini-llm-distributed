import torch

from minigpt.data import get_batch, get_dataset


def test_dataset_nonempty():
    data, vocab = get_dataset()
    assert len(data) > 100
    assert vocab > 1


def test_batch_shapes_and_determinism():
    data, _ = get_dataset()
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    x1, y1 = get_batch(data, 16, 8, generator=g1)
    x2, _ = get_batch(data, 16, 8, generator=g2)
    assert x1.shape == (8, 16)
    assert torch.equal(x1, x2)  # детерминизм при одном сиде
    assert torch.equal(y1[:, :-1], x1[:, 1:])  # y — сдвиг x на один токен
