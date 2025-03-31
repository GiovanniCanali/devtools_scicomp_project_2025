import torch
import torch.nn as nn
from torch.nn.functional import pad
import torch.nn.functional as F
from ...utils import compute_hippo, compute_dplr


class S4LowRankBlock(nn.Module):
    """
    TODO
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        method: str = "convolutional",
        dt: float = 0.1,
        hippo: bool = True,
    ):
        super().__init__()
        if method != "convolutional":
            raise NotImplementedError(
                "S4LowRankBlock supports only 'convolutional method"
            )
        A = compute_hippo(hidden_dim)
        # Define parameters
        self.B = torch.nn.Parameter(torch.rand(input_dim, hidden_dim))
        self.C_tilde = torch.nn.Parameter(torch.rand(input_dim, hidden_dim))
        if hippo:
            self.Lambda, self.P, self.Q = compute_dplr(A)
            self.Lambda = self.Lambda.unsqueeze(0).repeat(input_dim, 1, 1)
            self.P = self.P.repeat(input_dim, 1)
            self.Q = self.Q.repeat(input_dim, 1)
        else:
            self.Lambda = torch.rand(input_dim, 1, hidden_dim)
            self.P = torch.rand(input_dim, hidden_dim)
            self.Q = torch.rand(input_dim, hidden_dim)

        self.Lambda = torch.nn.Parameter(self.Lambda)
        self.P = torch.nn.Parameter(self.P)
        self.Q = torch.nn.Parameter(self.Q)

        self.register_buffer("omega", None)
        self.register_buffer("dt", torch.tensor([dt]))
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

    @staticmethod
    def _init_omega(L):
        return torch.exp(2j * torch.pi * torch.arange(L) / L)  # (L,)

    def _cauchy_dot(self, a0, a1, b0, b1, denominator):
        v00 = (a0 * b0).unsqueeze(1)
        v01 = (a0 * b1).unsqueeze(1)
        v10 = (a1 * b0).unsqueeze(1)
        v11 = (a1 * b1).unsqueeze(1)
        k00 = (v00 / denominator).sum(-1)
        k01 = (v01 / denominator).sum(-1)
        k10 = (v10 / denominator).sum(-1)
        k11 = (v11 / denominator).sum(-1)
        return k00, k01, k10, k11

    def compute_K(self, L):
        # Store the complex conjugates and other parameters
        a0, a1 = self.C_tilde.conj(), self.Q
        b0, b1 = self.B, self.P

        # Compute the denominator
        g = (2.0 / self.dt) * (
            (1.0 - self.omega) / (1.0 + self.omega)
        ).unsqueeze(-1)
        denominator = torch.cat(
            [g - self.Lambda[i : i + 1, :, :] for i in range(self.input_dim)],
            dim=0,
        )

        # Compute the Cauchy product
        k00, k01, k10, k11 = self._cauchy_dot(a0, a1, b0, b1, denominator)
        c = 2.0 / (1.0 + self.omega)

        # Compute the kernel in the frequency domain
        K_hat = c * (k00 - k01 * (1.0 + k11) * k10)

        # Return the kernel in the time domain
        return torch.fft.irfft(K_hat, n=L)

    def forward(self, x):
        # x: [B, L, input_dim]
        _, L, _ = x.shape
        if self.omega is None:
            self.omega = self._init_omega(L).to(x.device)
        x_reshaped = x.transpose(1, 2)  # [B, input_dim, L]

        # Compute kernel via Cauchy product
        K = self.compute_K(L)  # [input_dim, L]

        # Pad input and kernel to avoid circular convolution effects
        total_length = 2 * L  # Total length of the padded sequence
        x_padded = pad(x_reshaped, (0, L))  # [B, input_dim, total_length]
        K_padded = pad(K, (0, L))  # [input_dim, total_length]

        # Compute FFT of input and kernel
        x_fft = torch.fft.rfft(
            x_padded, dim=2
        )  # [B, input_dim, total_length//2+1]
        K_fft = torch.fft.rfft(
            K_padded, dim=1
        )  # [input_dim, total_length//2+1]

        # Element-wise multiplication in frequency domain
        K_fft = K_fft.unsqueeze(0)  # [1, input_dim, total_length//2+1]
        y_fft = x_fft * K_fft  # [B, input_dim, total_length//2+1]

        # IFFT back to time domain: [B, input_dim, total_length]
        y = torch.fft.irfft(y_fft, n=total_length, dim=2)
        y = y[:, :, :L]  # [B, input_dim, L]
        return y.transpose(1, 2)  # [B, L, input_dim]
