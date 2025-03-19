import torch
from torch.nn.functional import pad


class S4BaseBlock(torch.nn.Module):
    """
    Implementation of the S4 block.

    This block supports three forward pass methods: continuous, recurrent, and
    Fourier.

    - **Continuous**: It uses the block's continuous-time dynamics.
    - **Recurrent**: It applies discretized dynamics for sequential processing.
    - **Fourier**: It leverages the Fourier transform to compute convolutions.

    The block is defined by the following equations:

    .. math::
        \dot{h}(t) = Ah(t) + Bx(t),
        y(t) = C h(t),

    where :math:`h(t)` is the hidden state, :math:`x(t)` is the input,
    :math:`y(t)` is the output, :math:`A`is the hidden-to-hidden matrix,
    :math:`B` is the input-to-hidden matrix, and :math:`C` is the
    hidden-to-output matrix.

    .. seealso::
        **Original Reference**: Gu, A., Goel, K., and Re, G. (2021).
        "Efficiently Modeling Long Sequences with Structured State Spaces".
        arXiv:2111.00396.
        DOI: `<https://doi.org/10.48550/arXiv.2111.00396>_`.
    """

    def __new__(cls, method, **kwargs):
        """
        Creation of a new instance of the class. It dynamically sets the forward
        function based on the specified `method`.

        :param str method: The forward computation method.
            Available options are: `"continuous"`, `"recurrent"`, `"fourier"`.
        :param dict kwargs: Additional keyword arguments for customization.
        :raises ValueError: If an invalid `method` is provided.
        :return: A new instance of the class with the selected forward method.
        :rtype: S4BaseBlock
        """
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
        """
        Initialization of the S4 block.

        :param str method: The forward computation method.
            Available options are: `"continuous"`, `"recurrent"`, `"fourier"`.
        :param int hidden_dim: The hidden state dimension.
        :param float dt: The time step for discretization.
        :param bool hippo: Whether to use the HIPPO matrix for initialization.
        :param bool fixed: Whether to fix the hidden-to-hidden matrix :math:`A`.
        """
        super().__init__()

        # Dimensions
        self.hidden_dim = hidden_dim
        self.dt = dt

        # Initialization of A
        A = (
            self.hippo(hidden_dim)
            if hippo
            else torch.rand(self.hidden_dim, self.hidden_dim)
        )
        if fixed:
            self.register_buffer("A", A)
        else:
            self.A = torch.nn.Parameter(A)

        # Initialization of B and C
        self.B = torch.nn.Parameter(torch.rand(self.hidden_dim, 1))
        self.C = torch.nn.Parameter(torch.rand(1, self.hidden_dim))

        # Initialization of A_bar and B_bar for the discrete methods
        if method != "continuous":
            self.register_buffer("I", torch.eye(self.hidden_dim))
            self.register_buffer(
                "A_bar", torch.zeros(self.hidden_dim, self.hidden_dim)
            )
            self.register_buffer("B_bar", torch.zeros(self.hidden_dim, 1))

    def discretize(self):
        """
        Discretization of the continuous-time dynamics to obtain the matrices
        :math:`A_{bar}` and :math:`B_{bar}`.
        """
        tmp = self.I + self.A * self.dt / 2
        tmp2 = self.I - self.A * self.dt / 2
        self.A_bar = tmp2.inverse() @ tmp
        self.B_bar = tmp2.inverse() @ self.B * self.dt

    def _compute_K(self, L):
        """
        Computation of the convolution kernel K used in the Fourier method.
        K is defined as :math:`K = [C A^0 B, C A^1 B, ..., C A^{L-1} B]`.

        :param int L: The length of the sequence.
        :return: The convolution kernel :math:`K`.
        :rtype: torch.Tensor
        """
        K = torch.zeros(L, 1, 1)

        # Define A^0
        A_pow = torch.eye(self.hidden_dim)

        # Compute K
        for i in range(L - 1):
            K[i] = self.C @ A_pow @ self.B_bar
            A_pow = self.A_bar @ A_pow

        K[L - 1] = self.C @ A_pow @ self.B_bar
        return K

    def forward_continuous(self, x):
        """
        Forward pass using the continuous-time dynamics.

        :param torch.Tensor x: The input tensor with shape [B, L, 1].
        :return: The output tensor with shape [B, L, 1].
        :rtype: torch.Tensor

        .. note::
            Times for the hidden states h are shifted by one with respect to the
            input x. The hidden state h[0] is initialized with zeros.
        """
        # Permute the input tensor to [L, B, 1]
        sequence_length = x.shape[1]
        x = x.permute(1, 0)
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        # Initialize y and h
        y = torch.empty(x.shape[0], x.shape[1], 1)
        h = torch.empty(x.shape[0] + 1, x.shape[1], self.hidden_dim)

        # Compute the hidden states and the output
        h[0] = torch.zeros(x.shape[1], self.hidden_dim)
        for t in range(sequence_length):
            h[t + 1] = torch.einsum("ij,bj->bi", self.A, h[t]) + torch.einsum(
                "ij,bj->bi", self.B, x[t]
            )
            y[t] = torch.einsum("ij,bj->bi", self.C, h[t + 1])

        # Permute the output tensor to [B, L, 1]
        y = y.permute(1, 0, 2)
        return y

    def forward_recurrent(self, x):
        """
        Forward pass using the recurrent method.

        :param torch.Tensor x: The input tensor with shape [B, L, 1].
        :return: The output tensor with shape [B, L, 1].
        :rtype: torch.Tensor

        .. note::
            Times for the hidden states h are shifted by one with respect to the
            input x. The hidden state h[0] is initialized with zeros.
        """
        # Permute the input tensor to [L, B, 1]
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        sequence_length = x.shape[1]
        x = x.permute(1, 0, 2)

        # Initialize y and h
        y = torch.empty(x.shape[0], x.shape[1], 1)
        h = torch.empty(x.shape[0] + 1, x.shape[1], self.hidden_dim)

        # Discretize the continuous-time dynamics
        self.discretize()

        # Compute the hidden states and the output
        h[0] = torch.zeros(x.shape[1], self.hidden_dim)
        for t in range(sequence_length):
            h[t + 1] = torch.einsum(
                "ij,bj->bi", self.A_bar, h[t]
            ) + torch.einsum("ij,bj->bi", self.B_bar, x[t])
            y[t] = torch.einsum("ij,bj->bi", self.C, h[t + 1])

        # Permute the output tensor to [B, L, 1]
        y = y.permute(1, 0, 2)
        return y

    def forward_fourier(self, x):
        """
        Forward pass using the Fourier method.

        :param torch.Tensor x: The input tensor with shape [B, L, 1].
        :return: The output tensor with shape [B, L, 1].
        :rtype: torch.Tensor
        """
        # Discretize the continuous-time dynamics
        L = x.shape[1]
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        self.discretize()

        # Compute the convolution kernel K
        K = self._compute_K(L)

        # Apply zero-padding to avoid circular convolution effects
        x_padded = pad(x, (0, 0, 0, L - 1))

        # Compute the convolution using the Fourier transform
        x_fft = torch.fft.rfft(x_padded, dim=1)
        K_fft = torch.fft.rfft(
            pad(K, (0, 0, 0, 0, 0, x_padded.shape[1] - K.shape[0])), dim=0
        )
        y_fft = torch.einsum("bfi,foi->bfo", x_fft, K_fft)

        # Compute the inverse Fourier transform
        y = torch.fft.irfft(y_fft, n=x_padded.shape[1], dim=1)[:, :L, :]
        return y

    def change_forward(self, method):
        """
        Change the forward method of the block, depending on chosen `method`.

        :param str method: The forward computation method.
            Available options are: `"continuous"`, `"recurrent"`, `"fourier"`.
        :raises ValueError: If an invalid `method` is provided.
        """
        if method == "continuous":
            self.forward = self.forward_continuous
        elif method == "recurrent":
            self.forward = self.forward_recurrent
        elif method == "fourier":
            self.forward = self.forward_fourier
        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def hippo(N):
        """
        Definition of the HIPPO initialization for the hidden-to-hidden matrix.

        :param int N: The shape of the matrix.
        :return: A :math:`(N, N)` matrix initialized with the HIPPO method.
        :rtype: torch.Tensor
        """
        P = torch.sqrt(torch.arange(1, 2 * N, 2, dtype=torch.float32))
        A = 0.5 * (P[:, None] * P[None, :])
        A = torch.tril(A, diagonal=-1) - torch.diag(
            torch.arange(N, dtype=torch.float32)
        )
        return -A
