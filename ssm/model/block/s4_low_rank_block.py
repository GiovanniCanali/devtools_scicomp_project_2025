import torch
from .s4_block_interface import S4BlockInterface
from ...utils import compute_hippo, compute_dplr, initialize_dt


class S4LowRankBlock(S4BlockInterface):
    r"""
    Implementation of the low-rank S4 block.

    This block supports only the convolutional method for the forward pass.
    It allows an efficient computation of the convolutional kernel using the
    Cauchy product.

    .. seealso::
        **Original Reference**: Gu, A., Goel, K., and Re, G. (2021).
        "Efficiently Modeling Long Sequences with Structured State Spaces".
        arXiv:2111.00396.
        DOI: `<https://doi.org/10.48550/arXiv.2111.00396>_`.
    """

    def __init__(
        self,
        model_dim,
        hid_dim,
        method,
        dt_min=0.001,
        dt_max=0.1,
        hippo=True,
        **kwargs,
    ):
        """
        Initialization of the low-rank S4 block.

        :param int model_dim: The input dimension.
        :param int hid_dim: The hidden state dimension.
        :param str method: The forward computation method. Low-rank S4
            block only supports the convolutional method.
        :param float dt_min: Minimum time step for discretization. Default is
            `0.001`.
        :param float dt_max: Maximum time step for discretization. Default is
            `0.01`.
        :param bool hippo: Whether to use the HIPPO matrix for initialization.
            Default is `True`.
        :param dict kwargs: Additional arguments for the class constructor.
        """
        # Initialize matrices B and C
        B = torch.nn.Parameter(torch.rand(model_dim, hid_dim))
        C = torch.nn.Parameter(torch.rand(model_dim, hid_dim))

        # Initialize dt
        dt = initialize_dt(
            dim=model_dim,
            dt_max=dt_max,
            dt_min=dt_min,
            inverse_softplus=False,
        ).unsqueeze(-1)

        super().__init__(
            model_dim=model_dim,
            hid_dim=hid_dim,
            dt=dt,
            A=torch.empty((1)),
            B=B,
            C=C,
            method=method,
        )

        # Check on method
        if self.method != "convolutional":
            raise ValueError(
                "S4LowRankBlock only supports convolutional method,"
                f" got {self.method}"
            )

        # Initialize low-rank decomposition matrices
        if hippo:
            A = compute_hippo(hid_dim)
            self.Lambda, self.P, self.Q = compute_dplr(A)
            self.Lambda = self.Lambda.unsqueeze(0).repeat(model_dim, 1, 1)
            self.P = self.P.repeat(model_dim, 1)
            self.Q = self.Q.repeat(model_dim, 1)

        else:
            self.Lambda = torch.rand(model_dim, 1, hid_dim)
            self.P = torch.rand(model_dim, hid_dim)
            self.Q = torch.rand(model_dim, hid_dim)

        self.Lambda = torch.nn.Parameter(self.Lambda)
        self.P = torch.nn.Parameter(self.P)
        self.Q = torch.nn.Parameter(self.Q)

        # Initialize parameters
        self.register_buffer("omega", None)

    def forward_convolutional(self, x):
        """
        Forward pass.

        :param torch.Tensor x: The input tensor.
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        # Store the sequence length
        L = x.shape[1]

        # Compute the roots of unity for the FFT
        if self.omega is None:
            self.omega = self._compute_omega(L).to(x.device)

        # Reshape input to [B, D, L]
        x = x.transpose(1, 2)

        # Compute the kernel using the Cauchy product
        K = self._compute_K(L)

        # Pad input and kernel to avoid circular convolution effects
        x = torch.nn.functional.pad(x, (0, L))
        K = torch.nn.functional.pad(K, (0, L))

        # Compute FFT of input and kernel
        x_fft = torch.fft.rfft(x, dim=2)
        K_fft = torch.fft.rfft(K, dim=1).unsqueeze(0)

        # Element-wise multiplication in frequency domain
        y_fft = x_fft * K_fft

        # Inverse FFT
        y = torch.fft.irfft(y_fft, n=2 * L, dim=2)

        return y[:, :, :L].transpose(1, 2)

    def _compute_K(self, L):
        """
        Computation of the kernel K used in the convolutional method.

        :param int L: The length of the sequence.
        :return: The convolution kernel K.
        :rtype: torch.Tensor
        """
        dt = self.dt.clamp(min=1e-7, max=1e6)
        # Compute the matrices for the Cauchy product
        a0, a1 = self.C.conj(), self.Q
        b0, b1 = self.B, self.P

        # Compute the denominator for the Cauchy product
        g = (
            (2.0 / dt)
            * (1.0 - self.omega.unsqueeze(0))
            / (1.0 + self.omega.unsqueeze(0))
        ).unsqueeze(-1)
        denominator = g - self.Lambda

        # Compute the Cauchy product
        k00, k01, k10, k11 = self._cauchy_dot(a0, a1, b0, b1, denominator)
        c = 2.0 / (1.0 + self.omega)

        # Compute the kernel in the frequency domain
        K_hat = c * (k00 - k01 * (1.0 + k11) * k10)

        # Return the kernel in the time domain
        return torch.fft.irfft(K_hat, n=L)

    def _cauchy_dot(self, a0, a1, b0, b1, denominator):
        """
        Compute the Cauchy product of two sequences.

        :param torch.Tensor a0: Matrix A0.
        :param torch.Tensor a1: Matrix A1.
        :param torch.Tensor b0: Matrix B0.
        :param torch.Tensor b1: Matrix B1.
        :param torch.Tensor denominator: Denominator tensor.
        :return: The Cauchy product matrices.
        :rtype: tuple
        """
        v00 = (a0 * b0).unsqueeze(1)
        v01 = (a0 * b1).unsqueeze(1)
        v10 = (a1 * b0).unsqueeze(1)
        v11 = (a1 * b1).unsqueeze(1)
        k00 = (v00 / denominator).sum(-1)
        k01 = (v01 / denominator).sum(-1)
        k10 = (v10 / denominator).sum(-1)
        k11 = (v11 / denominator).sum(-1)

        return k00, k01, k10, k11

    @staticmethod
    def _compute_omega(L):
        """
        Compute the roots of unity for the FFT.

        :param int L: Length of the sequence.
        :return: The roots of unity.
        :rtype: torch.Tensor
        """
        return torch.exp(2j * torch.pi * torch.arange(L) / L)
