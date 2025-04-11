import torch
from ssm.model import S6

x = torch.rand(1000, 25, 5)


def test_s6_constructor():

    model = S6(
        input_dim=5,
        hid_dim=10,
        output_dim=2,
        n_layers=3,
        activation=torch.nn.ReLU,
    )


def test_s6_forward():

    model = S6(
        input_dim=5,
        hid_dim=10,
        output_dim=5,
        n_layers=3,
        activation=torch.nn.ReLU,
    )

    y = model.forward(x)
    assert y.shape == (1000, 25, 5)


def test_s6_backward():

    model = S6(
        input_dim=5,
        hid_dim=10,
        output_dim=5,
        n_layers=3,
        activation=torch.nn.ReLU,
    )

    y = model.forward(x.requires_grad_())
    _ = torch.mean(y).backward()
    assert x.grad.shape == x.shape
