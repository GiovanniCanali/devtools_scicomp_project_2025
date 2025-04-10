import pytest
import torch
from ssm.model.block import S4LowRankBlock

x = torch.rand(1000, 25, 5)


@pytest.mark.parametrize("hippo", [True, False])
def test_s4_low_rank_block_constructor(hippo):
    model = S4LowRankBlock(
        input_dim=5, hid_dim=10, hippo=hippo, method="convolutional"
    )

    assert model.P.shape == (5, 10)
    assert model.Q.shape == (5, 10)
    assert model.Lambda.shape == (5, 1, 10)

    # Check that "recurrent" method is not allowed
    with pytest.raises(ValueError):
        model = S4LowRankBlock(
            input_dim=5, hid_dim=10, hippo=hippo, method="recurrent"
        )

    # Invalid method
    with pytest.raises(ValueError):
        model = S4LowRankBlock(
            input_dim=5, hid_dim=10, hippo=hippo, method="invalid_method"
        )


@pytest.mark.parametrize("hippo", [True, False])
def test_s4_low_rank_block_forward(hippo):
    model = S4LowRankBlock(
        input_dim=5, hid_dim=10, hippo=hippo, method="convolutional"
    )
    y = model.forward(x)
    assert y.shape == (1000, 25, 5)


@pytest.mark.parametrize("hippo", [True, False])
def test_s4_low_rank_block_backward(hippo):
    model = S4LowRankBlock(
        input_dim=5, hid_dim=10, hippo=hippo, method="convolutional"
    )
    y = model.forward(x.requires_grad_())
    _ = torch.mean(y).backward()
    assert x._grad.shape == x.shape
