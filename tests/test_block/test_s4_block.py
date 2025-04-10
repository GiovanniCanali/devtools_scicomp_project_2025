import pytest
import torch
from ssm.model.block import S4BaseBlock
from ssm.model.block import S4LowRankBlock

x = torch.rand(1000, 25, 5)


@pytest.mark.parametrize("method", ["recurrent", "convolutional"])
@pytest.mark.parametrize("hippo", [True, False])
def test_s4_constructor(method, hippo):
    model = S4BaseBlock(method=method, hid_dim=10, hippo=hippo, input_dim=5)
    assert model.A.shape == (5, 10, 10)
    assert model.B.shape == (5, 10, 1)
    assert model.C.shape == (5, 1, 10)
    A_bar, B_bar = model._discretize()
    assert A_bar.shape == (5, 10, 10)
    assert B_bar.shape == (5, 10, 1)


@pytest.mark.parametrize("hippo", [True, False])
def test_s4_recurrent_forward(hippo):
    model = S4BaseBlock(
        method="recurrent", hid_dim=10, hippo=hippo, input_dim=5
    )
    y = model.forward(x)
    assert y.shape == (1000, 25, 5)


@pytest.mark.parametrize("hippo", [True, False])
def test_s4_convolutional_backward(hippo):
    model = S4BaseBlock(
        method="recurrent", hid_dim=10, hippo=hippo, input_dim=5
    )
    x.requires_grad = True
    y = model.forward(x)
    l = torch.mean(y)
    l.backward()
    assert x._grad.shape == x.shape


@pytest.mark.parametrize("hippo", [True, False])
def test_s4_convolutional_forward(hippo):
    model = S4BaseBlock(
        method="convolutional", hid_dim=10, hippo=hippo, input_dim=5
    )
    y = model.forward(x)
    assert y.shape == (1000, 25, 5)


@pytest.mark.parametrize("hippo", [True, False])
def test_s4_convolutional_backward(hippo):
    model = S4BaseBlock(
        method="convolutional", hid_dim=10, hippo=hippo, input_dim=5
    )
    x.requires_grad = True
    y = model.forward(x)
    l = torch.mean(y)
    l.backward()
    assert x._grad.shape == x.shape


@pytest.mark.parametrize("hippo", [True, False])
def test_s4_adv_constructor(hippo):
    model = S4LowRankBlock(
        method="convolutional", hid_dim=10, input_dim=5, hippo=hippo
    )
    assert model.P.shape == (5, 10)
    assert model.Q.shape == (5, 10)
    assert model.Lambda.shape == (5, 1, 10)


@pytest.mark.parametrize("hippo", [True, False])
def test_s4_adv_forward(hippo):
    model = S4LowRankBlock(
        method="convolutional", hid_dim=10, input_dim=5, hippo=hippo
    )
    y = model.forward(x)
    assert y.shape == (1000, 25, 5)


@pytest.mark.parametrize("hippo", [True, False])
def test_s4_adv_backward(hippo):
    model = S4LowRankBlock(
        method="convolutional", hid_dim=10, input_dim=5, hippo=hippo
    )
    y = model.forward(x)
    x.requires_grad = True
    y = model.forward(x)
    l = torch.mean(y)
    l.backward()
    assert x._grad.shape == x.shape
