import pytest
import torch
from ssm.model.block import S4BaseBlock

x = torch.rand(1000, 25, 5)


@pytest.mark.parametrize("hippo", [True, False])
def test_s4_recurrent(hippo):
    model = S4BaseBlock(
        method="recurrent", hidden_dim=10, hippo=hippo, input_dim=5
    )
    assert model.A.shape == (5, 10, 10)
    assert model.B.shape == (5, 10, 1)
    assert model.C.shape == (5, 1, 10)
    model.discretize()
    assert model.A_bar.shape == (5, 10, 10)
    assert model.B_bar.shape == (5, 10, 1)


@pytest.mark.parametrize("hippo", [True, False])
def test_s4_recurrent_forward(hippo):
    model = S4BaseBlock(
        method="recurrent", hidden_dim=10, hippo=hippo, input_dim=5
    )
    y = model.forward(x)
    assert y.shape == (1000, 25, 5)


@pytest.mark.parametrize("hippo", [True, False])
def test_s4_fourier_backward(hippo):
    model = S4BaseBlock(
        method="recurrent", hidden_dim=10, hippo=hippo, input_dim=5
    )
    x.requires_grad = True
    y = model.forward(x)
    l = torch.mean(y)
    l.backward()
    assert x._grad.shape == x.shape


@pytest.mark.parametrize("hippo", [True, False])
def test_s4_fourier(hippo):
    model = S4BaseBlock(
        method="fourier", hidden_dim=10, hippo=hippo, input_dim=5
    )
    assert model.A.shape == (5, 10, 10)
    assert model.B.shape == (5, 10, 1)
    assert model.C.shape == (5, 1, 10)
    model.discretize()
    assert model.A_bar.shape == (5, 10, 10)
    assert model.B_bar.shape == (5, 10, 1)


@pytest.mark.parametrize("hippo", [True, False])
def test_s4_fourier_forward(hippo):
    model = S4BaseBlock(
        method="fourier", hidden_dim=10, hippo=hippo, input_dim=5
    )
    y = model.forward(x)
    assert y.shape == (1000, 25, 5)


@pytest.mark.parametrize("hippo", [True, False])
def test_s4_fourier_backward(hippo):
    model = S4BaseBlock(
        method="fourier", hidden_dim=10, hippo=hippo, input_dim=5
    )
    x.requires_grad = True
    y = model.forward(x)
    l = torch.mean(y)
    l.backward()
    assert x._grad.shape == x.shape
