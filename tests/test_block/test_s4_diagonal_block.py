import pytest
import torch
from ssm.model.block import S4DBlock

x = torch.rand(20, 25, 5)
hid_dim = 10


@pytest.mark.parametrize("method", ["recurrent", "convolutional"])
@pytest.mark.parametrize("init_method", ["S4D-Inv", "S4D-Lin", "S4D-Quad"])
@pytest.mark.parametrize("real_random", [True, False])
@pytest.mark.parametrize("imag_random", [True, False])
@pytest.mark.parametrize("discretization", ["bilinear", "zoh"])
def test_s4_diagonal_block_constructor(
    method, init_method, discretization, imag_random, real_random
):

    model = S4DBlock(
        model_dim=x.shape[2],
        hid_dim=hid_dim,
        method=method,
        initialization=init_method,
        discretization=discretization,
        real_random=real_random,
        imag_random=imag_random,
    )

    assert model.A.shape == (x.shape[2], hid_dim)
    assert model.B.shape == (x.shape[2], hid_dim)
    assert model.C.shape == (x.shape[2], hid_dim)

    A_bar, B_bar = model._discretize()
    assert A_bar.shape == (x.shape[2], hid_dim)
    assert B_bar.shape == (x.shape[2], hid_dim)

    # Invalid method
    with pytest.raises(ValueError):
        model = S4DBlock(
            model_dim=x.shape[2], hid_dim=hid_dim, method="invalid_method"
        )

    # Invalid initialization
    with pytest.raises(ValueError):
        model = S4DBlock(
            model_dim=x.shape[2],
            hid_dim=hid_dim,
            method=method,
            initialization="inv_init",
        )

    # Invalid discretization
    with pytest.raises(ValueError):
        model = S4DBlock(
            model_dim=x.shape[2],
            hid_dim=hid_dim,
            method=method,
            discretization="inv_discr",
        )


@pytest.mark.parametrize("init_method", ["S4D-Inv", "S4D-Lin", "S4D-Quad"])
@pytest.mark.parametrize("real_random", [True, False])
@pytest.mark.parametrize("imag_random", [True, False])
@pytest.mark.parametrize("discretization", ["bilinear", "zoh"])
@pytest.mark.parametrize("method", ["recurrent", "convolutional"])
def test_s4_diagonal_block_forward(
    method, init_method, discretization, real_random, imag_random
):

    model = S4DBlock(
        model_dim=x.shape[2],
        hid_dim=hid_dim,
        method=method,
        initialization=init_method,
        discretization=discretization,
        real_random=real_random,
        imag_random=imag_random,
    )

    y = model.forward(x)
    assert y.shape == x.shape


@pytest.mark.parametrize("init_method", ["S4D-Inv", "S4D-Lin", "S4D-Quad"])
@pytest.mark.parametrize("real_random", [True, False])
@pytest.mark.parametrize("imag_random", [True, False])
@pytest.mark.parametrize("discretization", ["bilinear", "zoh"])
@pytest.mark.parametrize("method", ["recurrent", "convolutional"])
def test_s4_diagonal_block_backward(
    method, init_method, discretization, real_random, imag_random
):

    model = S4DBlock(
        model_dim=x.shape[2],
        hid_dim=hid_dim,
        method=method,
        initialization=init_method,
        discretization=discretization,
        real_random=real_random,
        imag_random=imag_random,
    )

    y = model.forward(x.requires_grad_())
    _ = torch.mean(y).backward()
    assert x.grad.shape == x.shape
