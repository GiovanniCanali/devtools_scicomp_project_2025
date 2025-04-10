import pytest
import torch
from ssm.model.block import S4DBlock

x = torch.rand(1000, 25, 5)


@pytest.mark.parametrize("method", ["recurrent", "convolutional"])
@pytest.mark.parametrize(
    "init_method", ["S4D-Inv", "S4D-Lin", "S4D-Quad", "real", "complex"]
)
@pytest.mark.parametrize("discretisation", ["bilinear", "zoh"])
def test_s4_diagonal_block_constructor(method, init_method, discretisation):
    model = S4DBlock(
        input_dim=5,
        hid_dim=10,
        method=method,
        initialization=init_method,
        discretization=discretisation,
    )

    assert model.A.shape == (5, 10)
    assert model.B.shape == (5, 10)
    assert model.C.shape == (5, 10)

    A_bar, B_bar = model._discretize()
    assert A_bar.shape == (5, 10)
    assert B_bar.shape == (5, 10)

    # Invalid method
    with pytest.raises(ValueError):
        model = S4DBlock(input_dim=5, hid_dim=10, method="invalid_method")

    # Invalid initialization
    with pytest.raises(ValueError):
        model = S4DBlock(
            input_dim=5, hid_dim=10, method=method, initialization="inv_init"
        )

    # Invalid discretization
    with pytest.raises(ValueError):
        model = S4DBlock(
            input_dim=5, hid_dim=10, method=method, discretization="inv_discr"
        )


@pytest.mark.parametrize(
    "init_method", ["S4D-Inv", "S4D-Lin", "S4D-Quad", "real", "complex"]
)
@pytest.mark.parametrize("discretisation", ["bilinear", "zoh"])
@pytest.mark.parametrize("method", ["recurrent", "convolutional"])
def test_s4_diagonal_block_forward(method, init_method, discretisation):
    model = S4DBlock(
        input_dim=5,
        hid_dim=10,
        method=method,
        initialization=init_method,
        discretization=discretisation,
    )
    y = model.forward(x)
    assert y.shape == (1000, 25, 5)


@pytest.mark.parametrize(
    "init_method", ["S4D-Inv", "S4D-Lin", "S4D-Quad", "real", "complex"]
)
@pytest.mark.parametrize("discretisation", ["bilinear", "zoh"])
@pytest.mark.parametrize("method", ["recurrent", "convolutional"])
def test_s4_diagonal_block_backward(method, init_method, discretisation):
    model = S4DBlock(
        input_dim=5,
        hid_dim=10,
        method=method,
        initialization=init_method,
        discretization=discretisation,
    )
    y = model.forward(x.requires_grad_())
    _ = torch.mean(y).backward()
    assert x._grad.shape == x.shape
