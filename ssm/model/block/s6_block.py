import math
import torch
from ...utils import compute_S4DReal, initialize_dt
from ...pscan import pscan


class DeltaNetwork(torch.nn.Module):
    """
    Implementation of the Delta Network used in S6.

    The Delta Network is a simple multi-layer perceptron that computes the time
    step size for the S6 block. The resulting time step is broadcasted to the
    input dimension.
    """

    def __init__(self, model_dim, dt_min, dt_max, dt_rank, dt_scale=1.0):
        """
        Initialization of the Delta Network.

        :param int model_dim: The input dimension.
        :param float dt_min: The minimum time step for discretization.
        :param float dt_max: The maximum time step for discretization.
        """
        super().__init__()
        self.model_dim = model_dim
        self.linear = torch.nn.Linear(model_dim, dt_rank, bias=True)
        self.activation = torch.nn.Softplus()

        # Initialize the time step dt
        dt = initialize_dt(
            dim=model_dim,
            dt_min=dt_min,
            dt_max=dt_max,
            inverse_softplus=True,
        )
        self.project = torch.nn.Linear(dt_rank, model_dim, bias=True)
        dt_init_std = dt_rank**-0.5 * dt_scale
        torch.nn.init.uniform_(self.project.weight, -dt_init_std, dt_init_std)
        with torch.no_grad():
            self.project.bias.copy_(dt)

    def forward(self, x):
        """
        Forward pass of the Delta Network.

        :param torch.Tensor x: The input tensor.
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        x = self.linear(x)
        return self.activation(self.project(x))


class S6Block(torch.nn.Module):
    r"""
    Implementation of the S6 block.

    This block is designed to efficiently model long sequences using selective
    state space models. Its selection mechanism allows it to focus on relevant
    parts of the input sequence, making it suitable for tasks such as selective
    copy.

    The output is computed in an efficient manner by leveraging the parallel
    scan algorithm.

    .. seealso::
        **Original Reference**: Gu, A., Dao, T. (2024).
        "Mamba: Linear-Time Sequence Modeling with Selective State Spaces".
        arXiv:2312.00752.
        DOI: `<https://arxiv.org/abs/2312.00752>_`.

        **Original Reference**: Heinsen, F., A. (2023)
        "Efficient Parallelization of a Ubiquitous Sequential Computation".
        arXiv:2311.06281.
        DOI: `<https://arxiv.org/abs/2311.06281>_`.
    """

    def __init__(
        self,
        model_dim,
        hid_dim,
        dt_min=0.001,
        dt_max=0.1,
        real_random=False,
        dt_rank=None,
        scan_type="parallel",
        **kwargs,
    ):
        """
        Initialization of the S6 block.

        :param int model_dim: The input dimension.
        :param int hid_dim: The hidden dimension.
        :param float dt_min: The minimum time step for discretization.
            Default is `0.001`.
        :param float dt_max: The maximum time step for discretization.
            Default is `0.01`.
        :param bool real_random: If `True`, the real part of the A matrix is
            initialized at random between 0 and 1. Default is `False`.
        :param int dt_rank: The rank of the time step. Default is `1`.
        :param str scan_type: The type of scan to use. Can be either
            "sequential" or "parallel". Default is "parallel".
        :param dict kwargs: Additional keyword arguments.
        """
        super().__init__()

        # Initialize parameters
        self.model_dim = model_dim
        self.hid_dim = hid_dim
        dt_rank = dt_rank if dt_rank is not None else max(model_dim // 16, 1)

        # Initialize the matrix A
        A = compute_S4DReal(hid_dim, real_random=real_random).unsqueeze(0)
        self.A = torch.nn.Parameter(
            A.repeat(model_dim, 1).unsqueeze(0).unsqueeze(0)
        )
        # Initialize the networks to compute matrices B and C
        self.linear = torch.nn.Linear(model_dim, hid_dim * 2)

        self.delta_net = DeltaNetwork(
            model_dim=model_dim, dt_min=dt_min, dt_max=dt_max, dt_rank=dt_rank
        )
        self.D = torch.nn.Parameter(torch.ones(model_dim))
        if scan_type == "sequential":
            self.scan = self.sequential_scan
        elif scan_type == "parallel":
            self.scan = pscan
        else:
            raise ValueError(
                f"Invalid scan type: {scan_type}. "
                "Choose either 'sequential' or 'parallel'."
            )

    def _discretize(self, A, B, dt):
        """
        Discretization of the continuous-time dynamics to obtain the matrices
        :math:`A_{bar}` and :math:`B_{bar}`.

        :param torch.Tensor A: The hidden-to-hidden matrix.
        :param torch.Tensor B: The input-to-hidden matrix.
        :param torch.Tensor dt: The time step for discretization.
        :return: The discretized matrices :math:`A_{bar}` and :math:`B_{bar}`.
        :rtype: tuple
        """
        tmp = A * dt.unsqueeze(-1)
        A_bar = torch.exp(tmp)
        delta_B = torch.matmul(dt.unsqueeze(-1), B.unsqueeze(2))
        B_bar = (A_bar - 1) * delta_B / tmp
        return A_bar, B_bar

    def forward(self, x):
        """
        Forward pass of the S6 block.

        :param torch.Tensor x: The input tensor.
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        # Compute matrices B and C
        B, C = self.linear(x).chunk(2, dim=-1)

        # Compute dt
        dt = self.delta_net(x).clamp(min=1e-7, max=1e6)

        # Discretize A and B
        A_bar, B_bar = self._discretize(self.A, B, dt)

        # Compute the second term for parallel scan
        B_barX = B_bar * x.unsqueeze(-1)

        # Perform parallel scan and compute the output
        scan = self.scan(A_bar, B_barX)
        return torch.sum(scan * C.unsqueeze(2), dim=-1) + self.D * x

    @staticmethod
    def sequential_scan(A, B):
        """
        Sequential scan of the input tensor using the given matrices A and B.

        :param torch.Tensor A: A tensor of shape (B, L, D, N).
        :param torch.Tensor B: Another tensor of shape (B, L, D, N).
        :return: The output tensor after sequential scan, of shape (B, L, D, N).
        :rtype: torch.Tensor
        """
        B_, L, D, N = A.shape
        h = B[:, 0]
        H_list = [h]

        for t in range(1, L):
            h = A[:, t] * h + B[:, t]
            H_list.append(h)

        return torch.stack(H_list, dim=1)
