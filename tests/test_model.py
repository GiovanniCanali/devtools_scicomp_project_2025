import pytest
import torch
from ssm.s4 import S4Base, S4Recurrent, S4Fourier, S4

x = torch.rand(1000, 25, 5)


def test_s4_base():
    with pytest.raises(TypeError):
        S4Base(latent_dim=10, input_dim=10)


def test_s4_base_continuous():
    model = S4(method="continuous", latent_dim=10, input_dim=5, output_dim=2)
    assert model.A.shape == (10, 10)
    assert model.B.shape == (10, 5)
    assert model.C.shape == (2, 10)
    assert not hasattr(model, "A_tilde")
    assert not hasattr(model, "B_tilde")


def test_s4_base_forward():
    model = S4(method="continuous", latent_dim=10, input_dim=5, output_dim=2)
    y = model.forward(x)
    assert y.shape == (1000, 25, 2)


def test_s4_recurrent():
    model = S4Recurrent(latent_dim=10, input_dim=5, output_dim=2)
    assert model.A.shape == (10, 10)
    assert model.B.shape == (10, 5)
    assert model.C.shape == (2, 10)
    model.discretize()
    assert model.A_tilde.shape == (10, 10)
    assert model.B_tilde.shape == (10, 5)


def test_s4_recurrent_forward():
    model = S4(method="recurrent", latent_dim=10, input_dim=5, output_dim=2)
    y = model.forward(x)
    assert y.shape == (1000, 25, 2)


def test_s4_fourier():
    model = S4Fourier(latent_dim=10, input_dim=5, output_dim=2)
    assert model.A.shape == (10, 10)
    assert model.B.shape == (10, 5)
    assert model.C.shape == (2, 10)
    model.discretize()
    assert model.A_tilde.shape == (10, 10)
    assert model.B_tilde.shape == (10, 5)


def test_s4_fourier_forward():
    model = S4(method="fourier", latent_dim=10, input_dim=5, output_dim=2)
    y = model.forward(x)
    assert y.shape == (1000, 25, 2)
    model_r = S4(method="recurrent", latent_dim=10, input_dim=5, output_dim=2)
    model_r.A = model.A
    model_r.B = model.B
    model_r.C = model.C
    y_r = model_r.forward(x)
    assert torch.allclose(y, y_r, atol=1e-5)
