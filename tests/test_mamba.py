import pytest
import torch
from ssm.model import Mamba

x = torch.randn(20, 25, 5)


@pytest.mark.parametrize("n_layers", [1, 4])
@pytest.mark.parametrize("ssm_type", ["S4", "S4D", "S4LowRank", "S6"])
@pytest.mark.parametrize("normalization", [True, False])
@pytest.mark.parametrize("method", ["convolutional", "recurrent"])
def test_mamba_constructor(n_layers, ssm_type, normalization, method):

    # Skip the test for S4LowRank with recurrent method
    if ssm_type == "S4LowRank" and method == "recurrent":
        return

    model = Mamba(
        n_layers=n_layers,
        model_dim=x.shape[2],
        expansion_factor=2,
        hid_dim=12,
        kernel_size=3,
        normalization=normalization,
        ssm_type=ssm_type,
        method=method,
    )

    with pytest.raises(KeyError):
        Mamba(n_layers=1)


@pytest.mark.parametrize("normalization", [True, False])
@pytest.mark.parametrize("ssm_type", ["S4", "S4D", "S4LowRank", "S6"])
@pytest.mark.parametrize("method", ["convolutional", "recurrent"])
def test_mamba_forward(normalization, ssm_type, method):

    # Skip the test for S4LowRank with recurrent method
    if ssm_type == "S4LowRank" and method == "recurrent":
        return

    model = Mamba(
        n_layers=1,
        model_dim=x.shape[2],
        expansion_factor=2,
        hid_dim=12,
        kernel_size=3,
        ssm_type=ssm_type,
        normalization=normalization,
        method=method,
    )

    y = model(x)
    assert y.shape == x.shape


@pytest.mark.parametrize("normalization", [True, False])
@pytest.mark.parametrize("ssm_type", ["S4", "S4D", "S4LowRank", "S6"])
@pytest.mark.parametrize("method", ["convolutional", "recurrent"])
def test_mamba_backward(ssm_type, normalization, method):

    # Skip the test for S4LowRank with recurrent method
    if ssm_type == "S4LowRank" and method == "recurrent":
        return

    model = Mamba(
        n_layers=1,
        model_dim=x.shape[2],
        expansion_factor=2,
        hid_dim=12,
        kernel_size=3,
        ssm_type=ssm_type,
        normalization=normalization,
        method=method,
    )

    y = model(x.requires_grad_())
    _ = torch.mean(y).backward()
    assert x.grad.shape == x.shape
