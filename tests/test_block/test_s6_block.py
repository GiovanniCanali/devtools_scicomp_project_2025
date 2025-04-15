import torch
from ssm.model.block import S6Block

x = torch.rand(20, 25, 5)
hid_dim = 10


def test_s6_block_constructor():

    model = S6Block(input_dim=x.shape[2], hid_dim=hid_dim)

    assert model.A.shape == (x.shape[2], hid_dim)
    assert hasattr(model, "linear_b")
    assert hasattr(model, "linear_c")
    assert hasattr(model, "delta_net")


def test_s6_block_forward():

    model = S6Block(input_dim=x.shape[2], hid_dim=hid_dim)
    y = model.forward(x)

    assert y.shape == x.shape


def test_s6_block_backward():

    model = S6Block(input_dim=x.shape[2], hid_dim=hid_dim)
    y = model.forward(x.requires_grad_())
    _ = torch.mean(y).backward()

    assert x.grad.shape == x.shape
