import pytest
import torch
from ssm.model.block import S6Block

x = torch.rand(20, 25, 5)
hid_dim = 10


@pytest.mark.parametrize("scan_type", ["parallel", "sequential"])
def test_s6_block_constructor(scan_type):

    model = S6Block(model_dim=x.shape[2], hid_dim=hid_dim)

    assert model.A.shape == (1, 1, x.shape[2], hid_dim)
    assert hasattr(model, "linear")
    assert hasattr(model, "delta_net")


def test_s6_block_forward():

    model = S6Block(model_dim=x.shape[2], hid_dim=hid_dim)
    y = model.forward(x)

    assert y.shape == x.shape


def test_s6_block_backward():

    model = S6Block(model_dim=x.shape[2], hid_dim=hid_dim)
    y = model.forward(x.requires_grad_())
    _ = torch.mean(y).backward()

    assert x.grad.shape == x.shape


def test_s6_squential_vs_parallel():
    # Test that the sequential and parallel versions of the S6Block produce the
    # same output
    model = S6Block(model_dim=x.shape[2], hid_dim=hid_dim)
    y_parallel = model.forward(x)
    model.scan = model.sequential_scan
    y_sequential = model.forward(x)
    assert torch.allclose(y_parallel, y_sequential, atol=1e-5, rtol=1e-5)
