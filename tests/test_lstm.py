import torch
from ssm.model import LSTM

x = torch.randn(20, 25, 5)

hid_dim = 10
n_layers = 2
dropout = 0.1


def test_lstm_constructor():

    LSTM(
        model_dim=x.shape[2],
        hidden_dim=hid_dim,
        n_layers=n_layers,
        dropout=dropout,
    )


def test_lstm_forward():

    model = LSTM(
        model_dim=x.shape[2],
        hidden_dim=hid_dim,
        n_layers=n_layers,
        dropout=dropout,
    )

    y = model(x)
    assert y.shape == x.shape


def test_lstm_backward():

    model = LSTM(
        model_dim=x.shape[2],
        hidden_dim=hid_dim,
        n_layers=n_layers,
        dropout=dropout,
    )

    y = model(x.requires_grad_())
    _ = torch.mean(y).backward()
    assert x.grad.shape == x.shape
