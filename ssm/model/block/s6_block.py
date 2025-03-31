import torch
from torch.nn import Softplus
from ...utils import compute_S4DReal


class DeltaNetwork(torch.nn.Module):
    def __init__(self, dt, input_dim, hidden_dim):
        super().__init__()
        self.dt = dt
        self.softplus = Softplus()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sc = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        delta = self.softplus(self.sc(x)) + self.dt  # [B, L, 1]
        return delta.repeat(1, 1, self.input_dim)  # [B, L, D]


class S6Block(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dt: float = 0.1,
        random_init: bool = False,
    ):
        super().__init__()

        # Dimensions
        self.input_dim = input_dim  # Input dimension
        self.hidden_dim = hidden_dim  # Hidden dimension

        if random_init:
            A = torch.rand(input_dim, hidden_dim)
        else:
            A = (
                compute_S4DReal(hidden_dim)
                .unsqueeze(0)
                .repeat(input_dim, 1)
                .clone()
            )

        # Set the A matrix and trainable parameter/Modules
        self.A = torch.nn.Parameter(A)
        self.dt = dt
        self.sb = torch.nn.Linear(input_dim, hidden_dim)
        self.sc = torch.nn.Linear(input_dim, hidden_dim)
        self.tau_delta = DeltaNetwork(
            dt=dt, input_dim=input_dim, hidden_dim=hidden_dim
        )

    def discretize(self, A, B, dt):
        delta_A = dt.unsqueeze(-1) * A
        A_bar = torch.exp(delta_A)
        A_inv = 1 / A_bar
        delta_B = torch.einsum("bld, bln -> bldn", dt, B)
        B_bar = A_inv * (A_bar - 1) * delta_B
        return A_bar, B_bar

    def forward(self, x):
        # Compute B
        B = self.sb(x)
        # Compute C
        C = self.sc(x)
        # Compute dt
        dt = self.tau_delta(x)

        # Discretize A and B -> A_bar, B_bar
        A_bar, B_bar = self.discretize(self.A, B, dt)  # Discretize A and B

        # Initialize the initial hidden state of size [B, input_dim, hidden_dim]
        h = torch.zeros(
            x.shape[0], self.input_dim, self.hidden_dim, device=x.device
        )
        # Initialize the output tensor of size [B, L, input_dim]
        y = torch.zeros((*x.shape[:-1], self.input_dim), device=x.device)

        # Loop over the sequence length
        for t in range(x.shape[1]):
            # Extract the x, A_bar, B_bar and C at time t
            x_t = x[:, t]
            A_bar_t = A_bar[:, t]
            B_bar_t = B_bar[:, t]
            C_t = C[:, t]
            # Update the hidden state
            h = A_bar_t * h + B_bar_t * x_t.unsqueeze(-1)
            # Compute the output
            y[:, t, :] = torch.sum(h * C_t.unsqueeze(1), dim=-1)
        return y
