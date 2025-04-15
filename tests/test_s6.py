import torch
from ssm.model import S6

x = torch.rand(20, 25, 5)
hid_dim = 10
out_dim = 2


def test_s6_constructor():

    model = S6(
        input_dim=x.shape[2],
        hid_dim=hid_dim,
        output_dim=out_dim,
        n_layers=3,
        activation=torch.nn.ReLU,
    )


def test_s6_forward():

    model = S6(
        input_dim=x.shape[2],
        hid_dim=hid_dim,
        output_dim=out_dim,
        n_layers=3,
        activation=torch.nn.ReLU,
    )

    y = model.forward(x)
    assert y.shape == (x.shape[0], x.shape[1], out_dim)


def test_s6_backward():

    model = S6(
        input_dim=x.shape[2],
        hid_dim=hid_dim,
        output_dim=out_dim,
        n_layers=3,
        activation=torch.nn.ReLU,
    )

    y = model.forward(x.requires_grad_())
    _ = torch.mean(y).backward()
    assert x.grad.shape == x.shape
