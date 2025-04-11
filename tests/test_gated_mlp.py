import pytest
import torch
from ssm.model import GatedMLP

x = torch.rand(1000, 25, 5)


@pytest.mark.parametrize("activation", ["silu", "swish"])
@pytest.mark.parametrize("beta", [1.0, 0.5, 2.0])
@pytest.mark.parametrize("n_layers", [1, 4])
def test_gated_mlp_constructor(activation, beta, n_layers):
    model = GatedMLP(
        input_dim=5,
        hid_dim=10,
        n_layers=n_layers,
        activation=activation,
        beta=beta,
    )
    assert len(model.gated_mlp_blocks) == n_layers

    # Invalid activation
    with pytest.raises(ValueError):
        GatedMLP(
            input_dim=5,
            hid_dim=10,
            n_layers=n_layers,
            activation="invalid_activation",
        )

    # Invalid beta
    with pytest.raises(ValueError):
        GatedMLP(
            input_dim=5,
            hid_dim=10,
            n_layers=n_layers,
            activation="swish",
            beta=-1.0,
        )


@pytest.mark.parametrize("activation", ["silu", "swish"])
@pytest.mark.parametrize("beta", [1.0, 0.5, 2.0])
@pytest.mark.parametrize("n_layers", [1, 4])
def test_gated_mlp_forward(activation, beta, n_layers):
    model = GatedMLP(
        input_dim=5,
        hid_dim=10,
        n_layers=n_layers,
        activation=activation,
        beta=beta,
    )

    y = model.forward(x)
    assert y.shape == (1000, 25, 5)


@pytest.mark.parametrize("activation", ["silu", "swish"])
@pytest.mark.parametrize("beta", [1.0, 0.5, 2.0])
@pytest.mark.parametrize("n_layers", [1, 4])
def test_gated_mlp_backward(activation, beta, n_layers):
    model = GatedMLP(
        input_dim=5,
        hid_dim=10,
        n_layers=n_layers,
        activation=activation,
        beta=beta,
    )

    y = model.forward(x.requires_grad_())
    _ = torch.mean(y).backward()
    assert x.grad.shape == x.shape
