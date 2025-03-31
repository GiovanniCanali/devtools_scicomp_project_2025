import torch
import pytest
from ssm.model import S4

x = torch.rand(1000, 25, 1)


@pytest.mark.parametrize("method", ["recurrent", "convolutional"])
@pytest.mark.parametrize("block_type", ["S4", "S4LowRank", "S4D"])
def test_s4_constructor(method, block_type):
    if block_type == "S4LowRank" and method == "recurrent":
        with pytest.raises(NotImplementedError):
            S4(
                method=method,
                block_type=block_type,
                input_dim=1,
                model_dim=5,
                output_dim=2,
                hidden_dim=10,
                n_layers=3,
            )
    else:
        S4(
            method=method,
            input_dim=1,
            block_type=block_type,
            model_dim=5,
            output_dim=2,
            hidden_dim=10,
            n_layers=3,
        )


@pytest.mark.parametrize("method", ["recurrent", "convolutional"])
@pytest.mark.parametrize("block_type", ["S4", "S4LowRank", "S4D"])
def test_s4_forward(method, block_type):
    if block_type == "S4LowRank" and method == "recurrent":
        return
    model = S4(
        method=method,
        block_type=block_type,
        input_dim=1,
        model_dim=5,
        output_dim=2,
        hidden_dim=10,
        n_layers=3,
    )
    y = model.forward(x)
    assert y.shape == (1000, 25, 2)


@pytest.mark.parametrize("method", ["recurrent", "convolutional"])
@pytest.mark.parametrize("block_type", ["S4", "S4LowRank", "S4D"])
def test_s4_backward(method, block_type):
    if block_type == "S4LowRank" and method == "recurrent":
        return
    model = S4(
        method=method,
        block_type=block_type,
        input_dim=1,
        model_dim=5,
        output_dim=2,
        hidden_dim=10,
        n_layers=3,
    )
    y = model.forward(x)
    x.requires_grad = True
    y = model.forward(x)
    l = torch.mean(y)
    l.backward()
    assert x._grad.shape == x.shape
