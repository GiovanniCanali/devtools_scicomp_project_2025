import pytest
import torch
from ssm.model import Transformer

x = torch.randn(20, 25, 4)

hid_dim = 10
n_layers = 2
dropout = 0.1


@pytest.mark.parametrize("heads", [2, 4])
@pytest.mark.parametrize("activation", ["gelu", "relu"])
def test_transformer_constructor(heads, activation):

    Transformer(
        model_dim=x.shape[2],
        hidden_dim=hid_dim,
        heads=heads,
        n_layers=n_layers,
        dropout=dropout,
        activation=activation,
    )

    # Invalid number of heads
    with pytest.raises(ValueError):
        Transformer(
            model_dim=x.shape[2],
            hidden_dim=hid_dim,
            heads=3,
            n_layers=n_layers,
            dropout=dropout,
            activation=activation,
        )


@pytest.mark.parametrize("heads", [2, 4])
@pytest.mark.parametrize("activation", ["gelu", "relu"])
def test_transformer_forward(heads, activation):

    model = Transformer(
        model_dim=x.shape[2],
        hidden_dim=hid_dim,
        heads=heads,
        n_layers=n_layers,
        dropout=dropout,
        activation=activation,
    )

    y = model(x)
    assert y.shape == x.shape


@pytest.mark.parametrize("heads", [2, 4])
@pytest.mark.parametrize("activation", ["gelu", "relu"])
def test_transformer_backward(heads, activation):

    model = Transformer(
        model_dim=x.shape[2],
        hidden_dim=hid_dim,
        heads=heads,
        n_layers=n_layers,
        dropout=dropout,
        activation=activation,
    )

    y = model(x.requires_grad_())
    _ = torch.mean(y).backward()
    assert x.grad.shape == x.shape
