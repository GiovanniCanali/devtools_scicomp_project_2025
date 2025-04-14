import pytest
import torch
from ssm.model.block import H3Block

x = torch.rand(1000, 25, 5)


@pytest.mark.parametrize("method", ["recurrent", "convolutional"])
@pytest.mark.parametrize("heads", [1, 5])
@pytest.mark.parametrize("discretisation", ["bilinear", "zoh"])
@pytest.mark.parametrize("init_method", ["S4D-Inv", "S4D-Lin", "S4D-Quad"])
def test_h3_block_constructor(method, heads, init_method, discretisation):
    model = H3Block(
        input_dim=5,
        hid_dim=10,
        method=method,
        heads=heads,
        initialization=init_method,
        discretization=discretisation,
    )

    # Invalid method
    with pytest.raises(ValueError):
        H3Block(
            input_dim=5,
            hid_dim=10,
            method="invalid_method",
            heads=heads,
            initialization=init_method,
            discretization=discretisation,
        )

    # Invalid heads
    with pytest.raises(ValueError):
        H3Block(
            input_dim=5,
            hid_dim=10,
            method=method,
            heads=3,
            initialization=init_method,
            discretization=discretisation,
        )


@pytest.mark.parametrize("method", ["recurrent", "convolutional"])
@pytest.mark.parametrize("heads", [1, 5])
@pytest.mark.parametrize("discretisation", ["bilinear", "zoh"])
@pytest.mark.parametrize("init_method", ["S4D-Inv", "S4D-Lin", "S4D-Quad"])
def test_h3_block_forward(method, heads, init_method, discretisation):
    model = H3Block(
        input_dim=5,
        hid_dim=10,
        method=method,
        heads=heads,
        initialization=init_method,
        discretization=discretisation,
    )

    y = model.forward(x)
    assert y.shape == (1000, 25, 5)


@pytest.mark.parametrize("method", ["recurrent", "convolutional"])
@pytest.mark.parametrize("heads", [1, 5])
@pytest.mark.parametrize("discretisation", ["bilinear", "zoh"])
@pytest.mark.parametrize("init_method", ["S4D-Inv", "S4D-Lin", "S4D-Quad"])
def test_h3_block_backward(method, heads, init_method, discretisation):
    model = H3Block(
        input_dim=5,
        hid_dim=10,
        method=method,
        heads=heads,
        initialization=init_method,
        discretization=discretisation,
    )

    y = model.forward(x.requires_grad_())
    _ = torch.mean(y).backward()
    assert x.grad.shape == x.shape
