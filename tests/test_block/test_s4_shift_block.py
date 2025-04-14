import pytest
import torch
from ssm.model.block import S4ShiftBlock

x = torch.rand(1000, 25, 5)


@pytest.mark.parametrize("method", ["recurrent", "convolutional"])
def test_s4_shift_block_constructor(method):
    model = S4ShiftBlock(
        input_dim=5,
        hid_dim=10,
        method=method,
    )

    assert model.A.requires_grad == False

    # Invalid method
    with pytest.raises(ValueError):
        model = S4ShiftBlock(input_dim=5, hid_dim=10, method="invalid_method")


@pytest.mark.parametrize("method", ["recurrent", "convolutional"])
def test_s4_shift_block_forward(method):
    model = S4ShiftBlock(
        input_dim=5,
        hid_dim=10,
        method=method,
    )

    y = model.forward(x)
    assert y.shape == (1000, 25, 5)


@pytest.mark.parametrize("method", ["recurrent", "convolutional"])
def test_s4_shift_block_backward(method):
    model = S4ShiftBlock(
        input_dim=5,
        hid_dim=10,
        method=method,
    )

    y = model.forward(x.requires_grad_())
    _ = torch.mean(y).backward()
    assert x.grad.shape == x.shape
