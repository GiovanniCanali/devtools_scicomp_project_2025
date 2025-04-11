import torch
from ...utils import compute_S4DReal


class DeltaNetwork(torch.nn.Module):
    """
    Implementation of the Delta Network used in S6.

    The Delta Network is a simple multi-layer perceptron that computes the time
    step size for the S6 block. The resulting time step is broadcasted to the
    input dimension.
    """

    def __init__(self, input_dim, dt):
        """
        Initialization of the Delta Network.

        :param int input_dim: The input dimension.
        :param float dt: The time step for discretization.
        """
        super().__init__()

        # Initialize the parameters
        self.dt = dt
        self.input_dim = input_dim

        # Define the layers
        self.activation = torch.nn.Softplus()
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        Forward pass of the Delta Network.

        :param torch.Tensor x: The input tensor.
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        delta = self.activation(self.linear(x)) + self.dt
        return delta.repeat(1, 1, self.input_dim)


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
        input_dim,
        hid_dim,
        dt=0.1,
        **kwargs,
    ):
        """
        Initialization of the S6 block.

        :param int input_dim: The input dimension.
        :param int hid_dim: The hidden dimension.
        :param float dt: The time step for discretization. Default is `0.1`.
        :param dict kwargs: Additional keyword arguments.
        """
        super().__init__()

        # Initialize parameters
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.dt = dt

        # Initialize the matrix A
        A = compute_S4DReal(hid_dim).unsqueeze(0).repeat(input_dim, 1)
        self.A = torch.nn.Parameter(A)

        # Initialize the networks to compute matrices B and C
        self.linear_b = torch.nn.Linear(input_dim, hid_dim)
        self.linear_c = torch.nn.Linear(input_dim, hid_dim)
        self.delta_net = DeltaNetwork(input_dim=input_dim, dt=dt)

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
        delta_B = torch.einsum("bld, bln -> bldn", dt, B)
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
        B = self.linear_b(x)
        C = self.linear_c(x)

        # Compute dt
        dt = self.delta_net(x)

        # Discretize A and B
        A_bar, B_bar = self._discretize(self.A, B, dt)

        # Compute the second term for parallel scan
        term2 = B_bar * x.unsqueeze(-1)

        # Perform parallel scan and compute the output
        scan = self._parallel_scan(A_bar, term2)
        return torch.sum(scan * C.unsqueeze(2), dim=-1)

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
