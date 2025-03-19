import torch
import torch.nn.functional as F


class S4BaseBlock(torch.nn.Module):
    def __new__(cls, method, **kwargs):
        instance = super().__new__(cls)
        if method == "continuous":
            instance.forward = instance.forward_continuous
        elif method == "recurrent":
            instance.forward = instance.forward_recurrent
        elif method == "fourier":
            instance.forward = instance.forward_fourier
        else:
            raise ValueError(f"Unknown method: {method}")
        return instance

    def __init__(
        self,
        method: str,
        hidden_dim: int,
        dt: float = 0.1,
        hippo: bool = False,
        fixed: bool = False,
    ):
        super().__init__()
        # Dimensions
        self.hidden_dim = hidden_dim
        self.dt = dt

        A = (
            self.hippo(hidden_dim)
            if hippo
            else torch.rand(self.hidden_dim, self.hidden_dim)
        )
        # Parameters
        if fixed:
            self.register_buffer("A", A)
        else:
            self.A = torch.nn.Parameter(A)
        self.B = torch.nn.Parameter(torch.rand(self.hidden_dim, 1))
        self.C = torch.nn.Parameter(torch.rand(1, self.hidden_dim))
        if method != "continuous":
            self.register_buffer("I", torch.eye(self.hidden_dim))
            self.register_buffer(
                "A_bar", torch.zeros(self.hidden_dim, self.hidden_dim)
            )
            self.register_buffer("B_bar", torch.zeros(self.hidden_dim, 1))

    @staticmethod
    def hippo(N):
        """
        Defining HIPPO matrix for the S4 block.
        :param int N: Shape of the matrix.
        :return: A matrix of shape (N, N) initialized according to the rules
            defined in the paper.
        :rtype: torch.Tensor
        """
        P = torch.sqrt(torch.arange(1, 2 * N, 2, dtype=torch.float32))
        A = 0.5 * (P[:, None] * P[None, :])
        A = torch.tril(A, diagonal=-1) - torch.diag(
            torch.arange(N, dtype=torch.float32)
        )
        return -A

    def discretize(self):
        tmp = self.I + self.A * self.dt / 2
        tmp2 = self.I - self.A * self.dt / 2
        self.A_bar = tmp2.inverse() @ tmp
        self.B_bar = tmp2.inverse() @ self.B * self.dt

    def _compute_K(self, L):
        """
        Compute the convolution kernel K = [CB, CAB, ..., CA^(L-1)B]
        """
        K = torch.zeros(L, 1, 1)
        A_pow = torch.eye(self.hidden_dim)  # A^0
        for i in range(L - 1):
            K[i] = self.C @ A_pow @ self.B_bar
            A_pow = self.A_bar @ A_pow  # A^(i+1)
        K[L - 1] = self.C @ A_pow @ self.B_bar
        return K

    def forward_fourier(self, x):
        L = x.shape[1]
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        self.discretize()

        K = self._compute_K(L)

        # Zero pad to avoid circular convolution effects
        x_padded = F.pad(x, (0, 0, 0, L - 1))

        x_fft = torch.fft.rfft(x_padded, dim=1)
        K_fft = torch.fft.rfft(
            F.pad(K, (0, 0, 0, 0, 0, x_padded.shape[1] - K.shape[0])), dim=0
        )
        y_fft = torch.einsum("bfi,foi->bfo", x_fft, K_fft)
        y = torch.fft.irfft(y_fft, n=x_padded.shape[1], dim=1)[:, :L, :]
        return y

    def forward_continuous(self, x):
        sequence_length = x.shape[1]
        x = x.permute(1, 0)  # [L, B]
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [L, B, 1]

        # create empty tensor for y and h
        y = torch.empty(x.shape[0], x.shape[1], 1)
        h = torch.empty(x.shape[0] + 1, x.shape[1], self.hidden_dim)

        # Notice that h times are shifted by one (i.e h[0] is time -1, the
        # initial condition)
        h[0] = torch.zeros(x.shape[1], self.hidden_dim)
        for t in range(sequence_length):
            h[t + 1] = torch.einsum("ij,bj->bi", self.A, h[t]) + torch.einsum(
                "ij,bj->bi", self.B, x[t]
            )
            y[t] = torch.einsum("ij,bj->bi", self.C, h[t + 1])
        y = y.permute(1, 0, 2)
        return y

    def forward_recurrent(self, x):
        # x has shape [B, L, 1]
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        sequence_length = x.shape[1]  # L
        x = x.permute(1, 0, 2)  # [L, B, 1]

        # create empty tensor for y and h
        y = torch.empty(x.shape[0], x.shape[1], 1)
        h = torch.empty(x.shape[0] + 1, x.shape[1], self.hidden_dim)

        # Apply the discretization of A and B and store them in A_bar and
        # B_bar
        self.discretize()

        # Notice that h times are shifted by one (i.e h[0] is time -1, the
        # initial condition)
        h[0] = torch.zeros(x.shape[1], self.hidden_dim)
        for t in range(sequence_length):
            h[t + 1] = torch.einsum(
                "ij,bj->bi", self.A_bar, h[t]
            ) + torch.einsum("ij,bj->bi", self.B_bar, x[t])
            y[t] = torch.einsum("ij,bj->bi", self.C, h[t + 1])
        y = y.permute(1, 0, 2)
        return y

    def change_forward(self, method):
        if method == "continuous":
            self.forward = self.forward_continuous
        elif method == "recurrent":
            self.forward = self.forward_recurrent
        elif method == "fourier":
            self.forward = self.forward_fourier
        else:
            raise ValueError(f"Unknown method: {method}")
