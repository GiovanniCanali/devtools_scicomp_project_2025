import pytest
import torch
from ssm.model.block import MambaBlock

x = torch.randn(20, 25, 5)


@pytest.mark.parametrize("ssm_type", ["S4", "S4D", "S4LowRank", "S6"])
@pytest.mark.parametrize("normalization", [True, False])
@pytest.mark.parametrize("method", ["convolutional", "recurrent"])
def test_mamba_block_constructor(ssm_type, normalization, method):

    # Skip the test for S4LowRank with recurrent method
    if ssm_type == "S4LowRank" and method == "recurrent":
        return

    model = MambaBlock(
        model_dim=x.shape[2],
        expansion_factor=2,
        kernel_size=3,
        hid_dim=12,
        normalization=normalization,
        ssm_type=ssm_type,
        method=method,
    )

    # Invalid ssm_type
    with pytest.raises(ValueError):
        MambaBlock(
            model_dim=x.shape[2],
            expansion_factor=2,
            kernel_size=3,
            hid_dim=12,
            normalization=normalization,
            ssm_type="invalid_ssm_type",
            method=method,
        )


@pytest.mark.parametrize("normalization", [True, False])
@pytest.mark.parametrize("ssm_type", ["S4D", "S4LowRank", "S6"])
@pytest.mark.parametrize("method", ["convolutional", "recurrent"])
def test_mamba_block_forward(normalization, ssm_type, method):

    # Skip the test for S4LowRank with recurrent method
    if ssm_type == "S4LowRank" and method == "recurrent":
        return

    model = MambaBlock(
        model_dim=x.shape[2],
        expansion_factor=2,
        kernel_size=3,
        hid_dim=12,
        normalization=normalization,
        ssm_type=ssm_type,
        method=method,
    )

    y = model(x)
    assert y.shape == x.shape


@pytest.mark.parametrize("normalization", [True, False])
@pytest.mark.parametrize("ssm_type", ["S4", "S4D", "S4LowRank", "S6"])
@pytest.mark.parametrize("method", ["convolutional", "recurrent"])
def test_mamba_block_backward(ssm_type, normalization, method):

    # Skip the test for S4LowRank with recurrent method
    if ssm_type == "S4LowRank" and method == "recurrent":
        return

    model = MambaBlock(
        model_dim=x.shape[2],
        expansion_factor=2,
        kernel_size=3,
        hid_dim=12,
        normalization=normalization,
        ssm_type=ssm_type,
        method=method,
    )

    y = model(x.requires_grad_())
    _ = torch.mean(y).backward()
    assert x.grad.shape == x.shape
