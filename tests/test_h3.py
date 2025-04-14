import pytest
import torch
from ssm.model import H3

x = torch.randn(15, 10, 5)


@pytest.mark.parametrize("method", ["convolutional", "recurrent"])
@pytest.mark.parametrize("heads", [1, 5])
@pytest.mark.parametrize("discretisation", ["bilinear", "zoh"])
@pytest.mark.parametrize("normalization", [True, False])
@pytest.mark.parametrize("init_method", ["S4D-Inv", "S4D-Lin", "S4D-Quad"])
def test_h3_constructor(
    method, heads, discretisation, normalization, init_method
):

    model = H3(
        input_dim=5,
        hid_dim=10,
        method=method,
        heads=heads,
        dt=0.1,
        initialization=init_method,
        discretization=discretisation,
        n_layers=3,
        normalization=normalization,
        activation=torch.nn.ReLU,
    )


@pytest.mark.parametrize("method", ["convolutional", "recurrent"])
@pytest.mark.parametrize("heads", [1, 5])
@pytest.mark.parametrize("discretisation", ["bilinear", "zoh"])
@pytest.mark.parametrize("normalization", [True, False])
@pytest.mark.parametrize("init_method", ["S4D-Inv", "S4D-Lin", "S4D-Quad"])
def test_h3_forward(method, heads, discretisation, normalization, init_method):

    model = H3(
        input_dim=5,
        hid_dim=10,
        method=method,
        heads=heads,
        dt=0.1,
        initialization=init_method,
        discretization=discretisation,
        n_layers=3,
        normalization=normalization,
        activation=torch.nn.ReLU,
    )

    y = model(x)
    assert y.shape == (15, 10, 5)


@pytest.mark.parametrize("method", ["convolutional", "recurrent"])
@pytest.mark.parametrize("heads", [1, 5])
@pytest.mark.parametrize("discretisation", ["bilinear", "zoh"])
@pytest.mark.parametrize("normalization", [True, False])
@pytest.mark.parametrize("init_method", ["S4D-Inv", "S4D-Lin", "S4D-Quad"])
def test_h3_backward(method, heads, discretisation, normalization, init_method):

    model = H3(
        input_dim=5,
        hid_dim=10,
        method=method,
        heads=heads,
        dt=0.1,
        initialization=init_method,
        discretization=discretisation,
        n_layers=3,
        normalization=normalization,
        activation=torch.nn.ReLU,
    )

    y = model(x.requires_grad_())
    _ = torch.mean(y).backward()
    assert x.grad.shape == x.shape
