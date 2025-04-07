import torch
from torch.nn.functional import pad
from ...utils import compute_hippo


class S4BaseBlock(torch.nn.Module):
    r"""
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
        elif method == "convolutional":
            instance.forward = instance.forward_convolutional
        else:
            raise ValueError(f"Unknown method: {method}")
        return instance

    def __init__(
        self,
        method: str,
        input_dim: int,
        hidden_dim: int,
        dt: float = 0.1,
        hippo: bool = False,
    ):
        """
        Initialization of the S4 block.

        :param str method: The forward computation method.
            Available options are: `"continuous"`, `"recurrent"`, `"fourier"`.
        :param int input_dim: The input dimension.
        :param int hidden_dim: The hidden state dimension.
        :param float dt: The time step for discretization.
        :param bool hippo: Whether to use the HIPPO matrix for initialization.
        :param bool fixed: Whether to fix the hidden-to-hidden matrix :math:`A`.
        """
        super().__init__()

        # Dimensions
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dt = dt

        # Initialize A for all channels
        if hippo:
            # For HIPPO, all channels share the same A matrix patterns
            A_base = compute_hippo(hidden_dim)
            A = (
                A_base.unsqueeze(0).expand(input_dim, -1, -1).clone()
            )  # [input_dim, hidden_dim, hidden_dim]
        else:
            # Otherwise, independent random initialization for each channel
            A = torch.rand(input_dim, hidden_dim, hidden_dim)

        self.A = torch.nn.Parameter(A)

        self.B = torch.nn.Parameter(torch.rand(input_dim, hidden_dim, 1))
        self.C = torch.nn.Parameter(torch.rand(input_dim, 1, hidden_dim))
        self.register_buffer(
            "I",
            torch.eye(hidden_dim).unsqueeze(0).expand(input_dim, -1, -1),
        )
        self.register_buffer(
            "A_bar", torch.zeros(input_dim, hidden_dim, hidden_dim)
        )
        self.register_buffer("B_bar", torch.zeros(input_dim, hidden_dim, 1))

    def discretize(self):
        """
        Discretization of the continuous-time dynamics to obtain the matrices
        :math:`A_{bar}` and :math:`B_{bar}`.
        """
        tmp = self.I + self.A * self.dt / 2
        tmp2 = self.I - self.A * self.dt / 2
        self.A_bar = tmp2.inverse() @ tmp
        self.B_bar = tmp2.inverse() @ self.B * self.dt

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
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Discretize dynamics for all channels at once
        self.discretize()

        h = torch.zeros(
            batch_size, self.input_dim, self.hidden_dim, device=x.device
        )  # [B, input_dim, hidden_dim]

        # [B, L, input_dim]
        y = torch.empty(batch_size, seq_len, self.input_dim, device=x.device)
        A_bar = self.A_bar.transpose(1, 2).unsqueeze(0)
        B_bar = self.B_bar.squeeze(-1).unsqueeze(0)
        C = self.C.transpose(1, 2).unsqueeze(0)
        # Iterate over time steps
        for t in range(seq_len):
            x_t = x[:, t, :]  # [B, input_dim]
            Ah = torch.matmul(h.unsqueeze(-2), A_bar).squeeze(-2)
            Bx = x_t.unsqueeze(-1) * B_bar
            h = Ah + Bx
            y_t = torch.matmul(h.unsqueeze(-2), C)
            y_t = y_t.squeeze(-2).squeeze(-1)
            y[:, t, :] = y_t

        # y = torch.stack(y, dim=1)
        return y

    def _compute_K(self, L):
        """
        Computation of the convolution kernel K used in the Fourier method.
        K is defined as :math:`K = [C A^0 B, C A^1 B, ..., C A^{L-1} B]`.

        :param int L: The length of the sequence.
        :return: The convolution kernel :math:`K`.
        :rtype: torch.Tensor
        """
        # Create kernel tensor: [input_dim, L]
        K = torch.zeros(self.input_dim, L, device=self.A.device)

        # Initialize A^0 for all channels (identity matrices)
        A_pow = self.I.clone()

        # Compute K for all channels at once
        for i in range(L):
            # Calculate C·A^i·B for all channels
            # Output shape: [input_dim, 1, 1]
            CAB = torch.bmm(torch.bmm(self.C, A_pow), self.B_bar)

            # Store in kernel tensor (squeeze to get scalar value per channel)
            K[:, i] = CAB.squeeze(-1).squeeze(-1)

            # Update A^i to A^(i+1)
            A_pow = torch.bmm(self.A_bar, A_pow)

        return K

    def forward_convolutional(self, x):
        """
        Forward pass using the Fourier method.

        :param torch.Tensor x: The input tensor with shape [B, L, 1].
        :return: The output tensor with shape [B, L, 1].
        :rtype: torch.Tensor
        """
        _, seq_len, _ = x.shape

        self.discretize()  # Discretize dynamics

        x_reshaped = x.transpose(1, 2)

        K = self._compute_K(seq_len)
        total_length = 2 * seq_len

        # Pad input and kernel to avoid circular convolution effects
        x_padded = pad(x_reshaped, (0, seq_len))
        K_padded = pad(K, (0, seq_len))

        # Compute FFT of input and kernel
        x_fft = torch.fft.rfft(x_padded, dim=2)
        K_fft = torch.fft.rfft(K_padded, dim=1)

        # Element-wise multiplication in frequency domain
        K_fft = K_fft.unsqueeze(0)  # [1, input_dim, total_length//2+1]
        y_fft = x_fft * K_fft  # [B, input_dim, total_length//2+1]

        # IFFT back to time domain: [B, input_dim, total_length]
        y = torch.fft.irfft(y_fft, n=total_length, dim=2)

        # Take only the first seq_len elements
        y = y[:, :, :seq_len]  # [B, input_dim, L]
        return y.transpose(1, 2)  # [B, L, input_dim]

    def change_forward(self, method):
        """Change the forward method."""
        if method == "recurrent":
            self.forward = self.forward_recurrent
        elif method == "convolutional":
            self.forward = self.forward_convolutional
        else:
            raise ValueError(f"Unknown method: {method}")
