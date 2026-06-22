import torch

from minigpt import GPT, GPTConfig


def test_forward_shapes_and_loss():
    cfg = GPTConfig(vocab_size=32, block_size=16, n_layer=2, n_head=2, n_embd=32)
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
    logits, loss = model(x, x)
    assert logits.shape == (4, cfg.block_size, cfg.vocab_size)
    assert loss.ndim == 0 and loss.item() > 0


def test_param_count_positive():
    assert GPT(GPTConfig()).num_params() > 0


def test_weight_tying_toggle():
    tied = GPT(GPTConfig(tie_weights=True))
    untied = GPT(GPTConfig(tie_weights=False))
    assert tied.tok.weight is tied.head.weight
    assert untied.tok.weight is not untied.head.weight
