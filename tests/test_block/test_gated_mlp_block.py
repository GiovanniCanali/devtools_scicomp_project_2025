import pytest
import torch
from ssm.model.block import GatedMLPBlock

x = torch.rand(1000, 25, 5)


@pytest.mark.parametrize("activation", ["silu", "swish"])
@pytest.mark.parametrize("beta", [1.0, 0.5, 2.0])
def test_gated_mlp_block_constructor(activation, beta):
    model = GatedMLPBlock(
        input_dim=5, hid_dim=10, activation=activation, beta=beta
    )

    # Invalid activation
    with pytest.raises(ValueError):
        GatedMLPBlock(input_dim=5, hid_dim=10, activation="invalid_activation")

    # Invalid beta
    with pytest.raises(ValueError):
        GatedMLPBlock(input_dim=5, hid_dim=10, activation="swish", beta=-1.0)


@pytest.mark.parametrize("activation", ["silu", "swish"])
@pytest.mark.parametrize("beta", [1.0, 0.5, 2.0])
def test_gated_mlp_block_forward(activation, beta):
    model = GatedMLPBlock(
        input_dim=5, hid_dim=10, activation=activation, beta=beta
    )

    y = model.forward(x)
    assert y.shape == (1000, 25, 5)


@pytest.mark.parametrize("activation", ["silu", "swish"])
@pytest.mark.parametrize("beta", [1.0, 0.5, 2.0])
def test_gated_mlp_block_backward(activation, beta):
    model = GatedMLPBlock(
        input_dim=5, hid_dim=10, activation=activation, beta=beta
    )

    y = model.forward(x.requires_grad_())
    _ = torch.mean(y).backward()
    assert x.grad.shape == x.shape
