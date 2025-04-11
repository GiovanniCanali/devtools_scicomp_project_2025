import torch
from ssm.model.block import S6Block

x = torch.rand(1000, 25, 5)


def test_s6_block_constructor():
    model = S6Block(input_dim=5, hid_dim=10)

    assert model.A.shape == (5, 10)
    assert hasattr(model, "linear_b")
    assert hasattr(model, "linear_c")
    assert hasattr(model, "delta_net")


def test_s6_block_forward():
    model = S6Block(input_dim=5, hid_dim=10)
    y = model.forward(x)

    assert y.shape == (1000, 25, 5)


def test_s6_block_backward():
    model = S6Block(input_dim=5, hid_dim=10)
    y = model.forward(x.requires_grad_())
    _ = torch.mean(y).backward()

    assert x.grad.shape == x.shape
