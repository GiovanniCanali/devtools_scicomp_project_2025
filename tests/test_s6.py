import pytest
import torch
from ssm.model import S6

x = torch.rand(20, 25, 5)
hid_dim = 10


@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("layer_norm", [True, False])
@pytest.mark.parametrize("scan_type", ["parallel", "sequential"])
def test_s6_constructor(residual, layer_norm, scan_type):

    model = S6(
        model_dim=x.shape[2],
        hid_dim=hid_dim,
        n_layers=3,
        activation=torch.nn.ReLU,
        residual=residual,
        layer_norm=layer_norm,
        scan_type=scan_type,
    )


@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("layer_norm", [True, False])
@pytest.mark.parametrize("scan_type", ["parallel", "sequential"])
def test_s6_forward(residual, layer_norm, scan_type):

    model = S6(
        model_dim=x.shape[2],
        hid_dim=hid_dim,
        n_layers=3,
        activation=torch.nn.ReLU,
        residual=residual,
        layer_norm=layer_norm,
        scan_type=scan_type,
    )

    y = model.forward(x)
    assert y.shape == x.shape


@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("layer_norm", [True, False])
@pytest.mark.parametrize("scan_type", ["parallel", "sequential"])
def test_s6_backward(residual, layer_norm, scan_type):

    model = S6(
        model_dim=x.shape[2],
        hid_dim=hid_dim,
        n_layers=3,
        activation=torch.nn.ReLU,
        residual=residual,
        layer_norm=layer_norm,
        scan_type=scan_type,
    )

    y = model.forward(x.requires_grad_())
    _ = torch.mean(y).backward()
    assert x.grad.shape == x.shape
