import pytest
import torch
from ssm.model.block import S4BaseBlock

x = torch.rand(1000, 25, 5)


@pytest.mark.parametrize("method", ["recurrent", "convolutional"])
@pytest.mark.parametrize("hippo", [True, False])
def test_s4_base_block_constructor(method, hippo):
    model = S4BaseBlock(input_dim=5, hid_dim=10, method=method, hippo=hippo)

    assert model.A.shape == (5, 10, 10)
    assert model.B.shape == (5, 10, 1)
    assert model.C.shape == (5, 1, 10)

    A_bar, B_bar = model._discretize()
    assert A_bar.shape == (5, 10, 10)
    assert B_bar.shape == (5, 10, 1)

    # Invalid method
    with pytest.raises(ValueError):
        model = S4BaseBlock(
            input_dim=5, hid_dim=10, hippo=hippo, method="invalid_method"
        )


@pytest.mark.parametrize("hippo", [True, False])
@pytest.mark.parametrize("method", ["recurrent", "convolutional"])
def test_s4_base_block_forward(hippo, method):
    model = S4BaseBlock(input_dim=5, hid_dim=10, method=method, hippo=hippo)
    y = model.forward(x)
    assert y.shape == (1000, 25, 5)


@pytest.mark.parametrize("hippo", [True, False])
@pytest.mark.parametrize("method", ["recurrent", "convolutional"])
def test_s4_base_block_backward(hippo, method):
    model = S4BaseBlock(input_dim=5, hid_dim=10, method=method, hippo=hippo)
    y = model.forward(x.requires_grad_())
    _ = torch.mean(y).backward()
    assert x.grad.shape == x.shape
