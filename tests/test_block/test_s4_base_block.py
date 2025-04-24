import pytest
import torch
from ssm.model.block import S4BaseBlock

x = torch.rand(20, 25, 5)
hid_dim = 10


@pytest.mark.parametrize("method", ["recurrent", "convolutional"])
@pytest.mark.parametrize("hippo", [True, False])
def test_s4_base_block_constructor(method, hippo):

    model = S4BaseBlock(
        model_dim=x.shape[2], hid_dim=hid_dim, method=method, hippo=hippo
    )

    assert model.A.shape == (x.shape[2], hid_dim, hid_dim)
    assert model.B.shape == (x.shape[2], hid_dim, 1)
    assert model.C.shape == (x.shape[2], 1, hid_dim)

    A_bar, B_bar = model._discretize()
    assert A_bar.shape == (x.shape[2], hid_dim, hid_dim)
    assert B_bar.shape == (x.shape[2], hid_dim, 1)

    # Invalid method
    with pytest.raises(ValueError):
        model = S4BaseBlock(
            model_dim=x.shape[2],
            hid_dim=hid_dim,
            hippo=hippo,
            method="invalid_method",
        )


@pytest.mark.parametrize("hippo", [True, False])
@pytest.mark.parametrize("method", ["recurrent", "convolutional"])
def test_s4_base_block_forward(hippo, method):

    model = S4BaseBlock(
        model_dim=x.shape[2], hid_dim=hid_dim, method=method, hippo=hippo
    )

    y = model.forward(x)
    assert y.shape == x.shape


@pytest.mark.parametrize("hippo", [True, False])
@pytest.mark.parametrize("method", ["recurrent", "convolutional"])
def test_s4_base_block_backward(hippo, method):

    model = S4BaseBlock(
        model_dim=x.shape[2], hid_dim=hid_dim, method=method, hippo=hippo
    )

    y = model.forward(x.requires_grad_())
    _ = torch.mean(y).backward()
    assert x.grad.shape == x.shape
