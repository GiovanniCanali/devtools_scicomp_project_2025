import pytest
import torch
from ssm.model import H3

x = torch.randn(20, 25, 5)
hid_dim = 10


@pytest.mark.parametrize("method", ["convolutional", "recurrent"])
@pytest.mark.parametrize("heads", [1, 5])
@pytest.mark.parametrize("discretisation", ["bilinear", "zoh"])
@pytest.mark.parametrize("normalization", [True, False])
@pytest.mark.parametrize("init_method", ["S4D-Inv", "S4D-Lin", "S4D-Real"])
def test_h3_constructor(
    method, heads, discretisation, normalization, init_method
):

    model = H3(
        input_dim=x.shape[2],
        hid_dim=hid_dim,
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
@pytest.mark.parametrize("init_method", ["S4D-Inv", "S4D-Lin", "S4D-Real"])
def test_h3_forward(method, heads, discretisation, normalization, init_method):

    model = H3(
        input_dim=x.shape[2],
        hid_dim=hid_dim,
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
    assert y.shape == x.shape


@pytest.mark.parametrize("method", ["convolutional", "recurrent"])
@pytest.mark.parametrize("heads", [1, 5])
@pytest.mark.parametrize("discretisation", ["bilinear", "zoh"])
@pytest.mark.parametrize("normalization", [True, False])
@pytest.mark.parametrize("init_method", ["S4D-Inv", "S4D-Lin", "S4D-Real"])
def test_h3_backward(method, heads, discretisation, normalization, init_method):

    model = H3(
        input_dim=x.shape[2],
        hid_dim=hid_dim,
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
