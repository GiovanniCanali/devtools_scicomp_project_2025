import pytest
import torch
from ssm.model.block import S4BaseBlock

x = torch.rand(1000, 25)


@pytest.mark.parametrize("hippo", [True, False])
def test_s4_base_continuous(hippo):
    model = S4BaseBlock(method="continuous", hidden_dim=10, hippo=hippo)
    assert model.A.shape == (10, 10)
    assert model.B.shape == (10, 1)
    assert model.C.shape == (1, 10)
    assert not hasattr(model, "A_bar")
    assert not hasattr(model, "B_bar")


@pytest.mark.parametrize("hippo", [True, False])
def test_s4_base_forward(hippo):
    model = S4BaseBlock(method="continuous", hidden_dim=10, hippo=hippo)
    y = model.forward(x)
    assert y.shape == (1000, 25, 1)


@pytest.mark.parametrize("hippo", [True, False])
def test_s4_recurrent(hippo):
    model = S4BaseBlock(method="recurrent", hidden_dim=10, hippo=hippo)
    assert model.A.shape == (10, 10)
    assert model.B.shape == (10, 1)
    assert model.C.shape == (1, 10)
    model.discretize()
    assert model.A_bar.shape == (10, 10)
    assert model.B_bar.shape == (10, 1)


@pytest.mark.parametrize("hippo", [True, False])
def test_s4_recurrent_forward(hippo):
    model = S4BaseBlock(method="recurrent", hidden_dim=10, hippo=hippo)
    y = model.forward(x)
    assert y.shape == (1000, 25, 1)


@pytest.mark.parametrize("hippo", [True, False])
def test_s4_fourier(hippo):
    model = S4BaseBlock(method="fourier", hidden_dim=10, hippo=hippo)
    assert model.A.shape == (10, 10)
    assert model.B.shape == (10, 1)
    assert model.C.shape == (1, 10)
    model.discretize()
    assert model.A_bar.shape == (10, 10)
    assert model.B_bar.shape == (10, 1)


@pytest.mark.parametrize("hippo", [True, False])
def test_s4_fourier_forward(hippo):
    model = S4BaseBlock(method="fourier", hidden_dim=10, hippo=hippo)
    y = model.forward(x)
    assert y.shape == (1000, 25, 1)
