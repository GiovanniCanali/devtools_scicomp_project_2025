from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F


class S4:
    def __new__(cls, method, **kwargs):
        if method == "continuous":
            return S4Continuous(**kwargs)
        if method == "recurrent":
            return S4Recurrent(**kwargs)
        if method == "fourier":
            return S4Fourier(**kwargs)
        raise ValueError(f"Unknown method: {method}")


class S4Base(torch.nn.Module, ABC):
    def __init__(
        self,
        latent_dim: int,
        input_dim: int,
        output_dim: int = None,
        dt: float = 0.1,
    ):
        super().__init__()
        # Dimensions
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.dt = dt

        # Parameters
        self.A = torch.nn.Parameter(
            torch.rand(self.latent_dim, self.latent_dim)
        )
        self.B = torch.nn.Parameter(torch.rand(self.latent_dim, self.input_dim))
        self.C = torch.nn.Parameter(
            torch.rand(self.output_dim, self.latent_dim)
        )

        self.register_buffer("I", torch.eye(self.latent_dim))
        self.register_buffer(
            "A_tilde", torch.zeros(self.latent_dim, self.latent_dim)
        )
        self.register_buffer(
            "B_tilde", torch.zeros(self.latent_dim, self.input_dim)
        )

    def discretize(self):
        tmp = self.I + self.A * self.dt / 2
        tmp2 = self.I - self.A * self.dt / 2
        self.A_tilde = torch.linalg.inv(tmp) @ tmp2
        self.B_tilde = torch.linalg.inv(tmp) @ self.B * self.dt

    @abstractmethod
    def forward(self, x):
        pass


class S4Continuous(S4Base):

    def __init__(self, latent_dim, input_dim, output_dim=None):
        super().__init__(latent_dim, input_dim, output_dim, dt=0.0)
        del self.A_tilde
        del self.B_tilde

    def forward(self, x):
        # x has shape [B, L, D]
        sequence_length = x.shape[1]
        x = x.permute(1, 0, 2)  # [L, B, D]
        # create empty tensor for y and h
        y = torch.empty(x.shape[0], x.shape[1], self.output_dim)
        h = torch.empty(x.shape[0] + 1, x.shape[1], self.latent_dim)

        # Notice that h times are shifted by one (i.e h[0] is time -1, the
        # initial condition)
        h[0] = torch.zeros(x.shape[1], self.latent_dim)
        for t in range(sequence_length):
            h[t + 1] = torch.einsum("ij,bj->bi", self.A, h[t]) + torch.einsum(
                "ij,bj->bi", self.B, x[t]
            )
            y[t] = torch.einsum("ij,bj->bi", self.C, h[t + 1])
        y = y.permute(1, 0, 2)
        return y


class S4Recurrent(S4Base):

    def forward(self, x):
        # x has shape [B, L, D]
        sequence_length = x.shape[1]  # L
        x = x.permute(1, 0, 2)  # [L, B, D]

        # create empty tensor for y and h
        y = torch.empty(x.shape[0], x.shape[1], self.output_dim)
        h = torch.empty(x.shape[0] + 1, x.shape[1], self.latent_dim)

        # Apply the discretization of A and B and store them in A_tilde and
        # B_tilde
        self.discretize()

        # Notice that h times are shifted by one (i.e h[0] is time -1, the
        # initial condition)
        h[0] = torch.zeros(x.shape[1], self.latent_dim)
        for t in range(sequence_length):
            h[t + 1] = torch.einsum(
                "ij,bj->bi", self.A_tilde, h[t]
            ) + torch.einsum("ij,bj->bi", self.B_tilde, x[t])
            y[t] = torch.einsum("ij,bj->bi", self.C, h[t + 1])
        y = y.permute(1, 0, 2)
        return y


class S4Fourier(S4Base):
    def _compute_K(self, L):
        """
        Compute the convolution kernel K = [CB, CAB, ..., CA^(L-1)B]
        """
        K = torch.zeros(L, self.output_dim, self.input_dim)
        A_pow = torch.eye(self.latent_dim)  # A^0
        for i in range(L - 1):
            K[i] = self.C @ A_pow @ self.B_tilde
            A_pow = self.A_tilde @ A_pow  # A^(i+1)
        K[L - 1] = self.C @ A_pow @ self.B_tilde
        return K

    def forward(self, x):
        self.discretize()
        B, L, D = x.shape

        K = self._compute_K(L)

        # Zero pad to avoid circular convolution effects
        x_padded = F.pad(x, (0, 0, 0, L - 1))

        x_fft = torch.fft.rfft(x_padded, dim=1)
        K_fft = torch.fft.rfft(
            F.pad(K, (0, 0, 0, 0, 0, x_padded.shape[1] - K.shape[0])), dim=0
        )
        y_fft = torch.einsum("bfi,foi->bfo", x_fft, K_fft)
        y = torch.fft.irfft(y_fft, n=x_padded.shape[1], dim=1)[:, :L]
        return y
