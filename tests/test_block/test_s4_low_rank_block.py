import pytest
import torch
from ssm.model.block import S4LowRankBlock

x = torch.rand(20, 25, 5)
hid_dim = 10


@pytest.mark.parametrize("hippo", [True, False])
def test_s4_low_rank_block_constructor(hippo):

    model = S4LowRankBlock(
        model_dim=x.shape[2],
        hid_dim=hid_dim,
        hippo=hippo,
        method="convolutional",
    )

    assert model.P.shape == (x.shape[2], hid_dim)
    assert model.Q.shape == (x.shape[2], hid_dim)
    assert model.Lambda.shape == (x.shape[2], 1, hid_dim)

    # Check that "recurrent" method is not allowed
    with pytest.raises(ValueError):
        model = S4LowRankBlock(
            model_dim=x.shape[2],
            hid_dim=hid_dim,
            hippo=hippo,
            method="recurrent",
        )

    # Invalid method
    with pytest.raises(ValueError):
        model = S4LowRankBlock(
            model_dim=x.shape[2],
            hid_dim=hid_dim,
            hippo=hippo,
            method="invalid_method",
        )


@pytest.mark.parametrize("hippo", [True, False])
def test_s4_low_rank_block_forward(hippo):

    model = S4LowRankBlock(
        model_dim=x.shape[2],
        hid_dim=hid_dim,
        hippo=hippo,
        method="convolutional",
    )

    y = model.forward(x)
    assert y.shape == x.shape


@pytest.mark.parametrize("hippo", [True, False])
def test_s4_low_rank_block_backward(hippo):

    model = S4LowRankBlock(
        model_dim=x.shape[2],
        hid_dim=hid_dim,
        hippo=hippo,
        method="convolutional",
    )

    y = model.forward(x.requires_grad_())
    _ = torch.mean(y).backward()
    assert x.grad.shape == x.shape
