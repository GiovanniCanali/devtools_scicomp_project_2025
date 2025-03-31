import torch
import pytest
from ssm.model.block import S6Block

x = torch.rand(1000, 25, 5)


@pytest.mark.parametrize("random_init", [True, False])
def test_s6_constructor(random_init):
    model = S6Block(hidden_dim=10, input_dim=5, random_init=random_init)
    assert model.A.shape == (5, 10)
    assert hasattr(model, "sb")
    assert hasattr(model, "sc")
    assert hasattr(model, "tau_delta")


@pytest.mark.parametrize("random_init", [True, False])
def test_s6_forward(random_init):
    model = S6Block(hidden_dim=10, input_dim=5, random_init=random_init)
    y = model.forward(x)
    assert y.shape == (1000, 25, 5)


@pytest.mark.parametrize("random_init", [True, False])
def test_s6_backward(random_init):
    model = S6Block(hidden_dim=10, input_dim=5, random_init=random_init)
    x.requires_grad = True
    y = model.forward(x)
    l = torch.mean(y)
    l.backward()
    assert x._grad.shape == x.shape
