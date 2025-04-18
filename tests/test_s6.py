import pytest
import torch
from ssm.model import S6

x = torch.rand(20, 25, 5)
hid_dim = 10
out_dim = 2


@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("layer_norm", [True, False])
def test_s6_constructor(residual, layer_norm):

    model = S6(
        input_dim=x.shape[2],
        hid_dim=hid_dim,
        output_dim=out_dim,
        n_layers=3,
        activation=torch.nn.ReLU,
        residual=residual,
        layer_norm=layer_norm,
    )


@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("layer_norm", [True, False])
def test_s6_forward(residual, layer_norm):

    model = S6(
        input_dim=x.shape[2],
        hid_dim=hid_dim,
        output_dim=out_dim,
        n_layers=3,
        activation=torch.nn.ReLU,
        residual=residual,
        layer_norm=layer_norm,
    )

    y = model.forward(x)
    assert y.shape == (x.shape[0], x.shape[1], out_dim)


@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("layer_norm", [True, False])
def test_s6_backward(residual, layer_norm):

    model = S6(
        input_dim=x.shape[2],
        hid_dim=hid_dim,
        output_dim=out_dim,
        n_layers=3,
        activation=torch.nn.ReLU,
        residual=residual,
        layer_norm=layer_norm,
    )

    y = model.forward(x.requires_grad_())
    _ = torch.mean(y).backward()
    assert x.grad.shape == x.shape
