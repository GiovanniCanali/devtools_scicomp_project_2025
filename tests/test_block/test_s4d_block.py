import pytest
import torch
from ssm.model.block import S4DBlock

x = torch.rand(1000, 25, 5)


@pytest.mark.parametrize("method", ["recurrent", "convolutional"])
@pytest.mark.parametrize(
    "init_method", ["S4D-Inv", "S4D-Lin", "S4D-Quad", "real", "complex"]
)
@pytest.mark.parametrize("discretisation", ["bilinear", "zoh"])
def test_s4d_constructor(method, init_method, discretisation):
    model = S4DBlock(
        method=method,
        hid_dim=10,
        input_dim=5,
        discretization=discretisation,
        initialization=init_method,
    )
    assert model.A.shape == (5, 10)
    assert model.B.shape == (5, 10)
    assert model.C.shape == (5, 10)
    model._discretize()
    assert model.A_bar.shape == (5, 10)
    assert model.B_bar.shape == (5, 10)


@pytest.mark.parametrize(
    "init_method", ["S4D-Inv", "S4D-Lin", "S4D-Quad", "real", "complex"]
)
@pytest.mark.parametrize("discretisation", ["bilinear", "zoh"])
def test_s4d_recurrent_forward(init_method, discretisation):
    model = S4DBlock(
        method="recurrent",
        hid_dim=10,
        input_dim=5,
        initialization=init_method,
        discretization=discretisation,
    )
    y = model.forward(x)
    assert y.shape == (1000, 25, 5)


@pytest.mark.parametrize(
    "init_method", ["S4D-Inv", "S4D-Lin", "S4D-Quad", "real", "complex"]
)
@pytest.mark.parametrize("discretisation", ["bilinear", "zoh"])
def test_s4d_convolutional_backward(init_method, discretisation):
    model = S4DBlock(
        method="recurrent",
        hid_dim=10,
        input_dim=5,
        initialization=init_method,
        discretization=discretisation,
    )
    x.requires_grad = True
    y = model.forward(x)
    l = torch.mean(y)
    l.backward()
    assert x._grad.shape == x.shape


@pytest.mark.parametrize(
    "init_method", ["S4D-Inv", "S4D-Lin", "S4D-Quad", "real", "complex"]
)
@pytest.mark.parametrize("discretisation", ["bilinear", "zoh"])
def test_s4d_convolutional_forward(init_method, discretisation):
    model = S4DBlock(
        method="convolutional",
        hid_dim=10,
        input_dim=5,
        initialization=init_method,
        discretization=discretisation,
    )
    y = model.forward(x)
    assert y.shape == (1000, 25, 5)


@pytest.mark.parametrize(
    "init_method", ["S4D-Inv", "S4D-Lin", "S4D-Quad", "real", "complex"]
)
@pytest.mark.parametrize("discretisation", ["bilinear", "zoh"])
def test_s4_convolutional_backward(init_method, discretisation):
    model = S4DBlock(
        method="convolutional",
        hid_dim=10,
        input_dim=5,
        initialization=init_method,
        discretization=discretisation,
    )
    x.requires_grad = True
    y = model.forward(x)
    l = torch.mean(y)
    l.backward()
    assert x._grad.shape == x.shape
