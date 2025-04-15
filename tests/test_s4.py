import torch
import pytest
from ssm.model import S4

x = torch.rand(20, 25, 5)
hid_dim = 10
out_dim = 2


@pytest.mark.parametrize("method", ["recurrent", "convolutional"])
@pytest.mark.parametrize("block_type", ["S4", "S4LowRank", "S4D"])
def test_s4_constructor(method, block_type):

    # Skip the test for S4LowRank with recurrent method
    if block_type == "S4LowRank" and method == "recurrent":
        return

    model = S4(
        input_dim=x.shape[2],
        hid_dim=hid_dim,
        output_dim=out_dim,
        method=method,
        n_layers=3,
        block_type=block_type,
        activation=torch.nn.ReLU,
    )

    assert model.block_type == block_type

    # Invalid block type
    with pytest.raises(ValueError):
        S4(
            input_dim=x.shape[2],
            hid_dim=hid_dim,
            output_dim=out_dim,
            method=method,
            n_layers=3,
            block_type="invalid_block_type",
            activation=torch.nn.ReLU,
        )


@pytest.mark.parametrize("method", ["recurrent", "convolutional"])
@pytest.mark.parametrize("block_type", ["S4", "S4LowRank", "S4D"])
def test_s4_forward(method, block_type):

    # Skip the test for S4LowRank with recurrent method
    if block_type == "S4LowRank" and method == "recurrent":
        return

    model = S4(
        input_dim=x.shape[2],
        hid_dim=hid_dim,
        output_dim=out_dim,
        method=method,
        n_layers=3,
        block_type=block_type,
        activation=torch.nn.ReLU,
    )

    y = model.forward(x)
    assert y.shape == (x.shape[0], x.shape[1], out_dim)


@pytest.mark.parametrize("method", ["recurrent", "convolutional"])
@pytest.mark.parametrize("block_type", ["S4", "S4LowRank", "S4D"])
def test_s4_backward(method, block_type):

    # Skip the test for S4LowRank with recurrent method
    if block_type == "S4LowRank" and method == "recurrent":
        return

    model = S4(
        input_dim=x.shape[2],
        hid_dim=hid_dim,
        output_dim=out_dim,
        method=method,
        n_layers=3,
        block_type=block_type,
        activation=torch.nn.ReLU,
    )

    y = model.forward(x.requires_grad_())
    _ = torch.mean(y).backward()
    assert x.grad.shape == x.shape
