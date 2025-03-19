import pytest
import torch
from ssm.model.block import S4BaseBlock

x = torch.rand(1000, 25) + 1


# def test_s4_base():
#     with pytest.raises(TypeError):
#         S4Base(hidden_dim=10, input_dim=10)


def test_s4_base_continuous():
    model = S4BaseBlock(method="continuous", hidden_dim=10)
    assert model.A.shape == (10, 10)
    assert model.B.shape == (10, 1)
    assert model.C.shape == (1, 10)
    assert not hasattr(model, "A_bar")
    assert not hasattr(model, "B_bar")


def test_s4_base_forward():
    model = S4BaseBlock(method="continuous", hidden_dim=10)
    y = model.forward(x)
    assert y.shape == (1000, 25, 1)


def test_s4_recurrent():
    model = S4BaseBlock(method="recurrent", hidden_dim=10)
    assert model.A.shape == (10, 10)
    assert model.B.shape == (10, 1)
    assert model.C.shape == (1, 10)
    model.discretize()
    assert model.A_bar.shape == (10, 10)
    assert model.B_bar.shape == (10, 1)


def test_s4_recurrent_forward():
    model = S4BaseBlock(method="recurrent", hidden_dim=10)
    y = model.forward(x)
    assert y.shape == (1000, 25, 1)


def test_s4_fourier():
    model = S4BaseBlock(method="fourier", hidden_dim=10)
    assert model.A.shape == (10, 10)
    assert model.B.shape == (10, 1)
    assert model.C.shape == (1, 10)
    model.discretize()
    assert model.A_bar.shape == (10, 10)
    assert model.B_bar.shape == (10, 1)


def test_s4_fourier_forward():
    model = S4BaseBlock(method="fourier", hidden_dim=10, hippo=True)
    y = model.forward(x)
    assert y.shape == (1000, 25, 1)
