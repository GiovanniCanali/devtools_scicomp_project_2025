import math
import torch
from ...utils import compute_S4DReal, initialize_dt


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
        B_bar = (A_bar - 1) * delta_B / (tmp + 1e-6)
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
        term2 = B_bar * x.unsqueeze(-1)

        # Perform parallel scan and compute the output
        scan = self._parallel_scan(A_bar, term2)
        return torch.sum(scan * C.unsqueeze(2), dim=-1) + self.D * x

    @staticmethod
    def _parallel_scan(a, b):
        """
        Compute in an efficient way sequences of the form:

        .. math::
            x_t = a_t x_{t-1} + b_t

        :param torch.Tensor a: The first tensor.
        :param torch.Tensor b: The second tensor.
        :return: The result of the parallel scan.
        :rtype: torch.Tensor
        """

        def _complex_log(input_):
            """
            Compute the complex logarithm of a tensor.

            :param torch.Tensor input_: The input tensor.
            :return: The complex logarithm of the input tensor.
            :rtype: torch.Tensor
            """
            # Compute real and imaginary parts
            eps = torch.ones_like(input_) * 1e-6
            real = input_.abs().maximum(eps).log()
            imag = (input_ < 0).to(input_.dtype) * torch.pi

            return torch.complex(real, imag)

        # Compute the complex logarithm of a and b
        log_a = _complex_log(a)
        log_b = _complex_log(b)

        # Compute the cumulative sum over the sequence length L
        a_star = torch.cumsum(log_a, dim=1)

        # Padding over the sequence length L
        a_star = torch.nn.functional.pad(a_star, (0, 0, 0, 0, 1, 0))

        # Compute the logcumsumexp over the sequence length L
        tmp = torch.logcumsumexp(log_b - a_star[:, 1:], dim=1)

        return torch.exp(a_star[:, 1:] + tmp).real
