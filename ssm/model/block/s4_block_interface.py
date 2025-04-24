from abc import ABC, abstractmethod
import torch


class S4BlockInterface(torch.nn.Module, ABC):
    r"""
    Implementation of the S4 block interface. Every S4 block should inherit
    from this interface and implement the required methods.

    This block supports two forward pass methods: recurrent, and convolutional.

    - **Recurrent**: It applies discretized dynamics for sequential processing.
    - **Convolutional**: It uses the Fourier transform to compute convolutions.

    The block is defined by the following equations:

    .. math::
        \dot{h}(t) = Ah(t) + Bx(t),
        y(t) = Ch(t),

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

        :param str method: The forward computation method. Available options
            are: recurrent, convolutional.
        :param dict kwargs: Additional arguments for the class constructor.
        :raises ValueError: If an invalid `method` is provided.
        :return: A new instance of the class with the selected forward method.
        :rtype: S4BlockInterface
        """
        instance = super().__new__(cls)
        if method == "recurrent":
            instance.forward = instance.forward_recurrent
        elif method == "convolutional":
            instance.forward = instance.forward_convolutional
        else:
            raise ValueError(f"Unknown method: {method}")
        return instance

    def __init__(self, model_dim, hid_dim, dt, A, B, C, method):
        """
        Initialization of the S4 block interface.

        :param int model_dim: The input dimension.
        :param int hid_dim: The hidden state dimension.
        :param float dt: The time step for discretization.
        :param torch.Tensor A: The hidden-to-hidden matrix.
        :param torch.Tensor B: The input-to-hidden matrix.
        :param torch.Tensor C: The hidden-to-output matrix.
        :param str method: The forward computation method. Available options
            are: recurrent, convolutional.
        :raises ValueError: If an invalid `method` is provided.
        """
        super().__init__()

        # Initialize parameters
        self.model_dim = model_dim
        self.hid_dim = hid_dim

        # Check method
        if method not in ["recurrent", "convolutional"]:
            raise ValueError(f"Unknown method: {method}")
        self.method = method

        # Initialize matrices A, B, C and the time step dt
        self.A, self.B, self.C, self.D, self.dt = (
            torch.nn.Parameter(A),
            torch.nn.Parameter(B),
            torch.nn.Parameter(C),
            torch.nn.Parameter(torch.rand(1, 1, model_dim)),
            torch.nn.Parameter(dt),
        )

    def forward_recurrent(self, x):
        """
        Forward pass using the recurrent method.

        :param torch.Tensor x: The input tensor .
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        A_bar, B_bar = self._discretize()

        # Store the batch size and the sequence length
        B, L = x.shape[0], x.shape[1]

        # Initialize the output tensor
        y = torch.empty(B, L, self.model_dim, device=x.device)

        # Initialize initial hidden state
        h = torch.zeros(B, self.model_dim, self.hid_dim, device=x.device)

        A_bar, B_bar, C = self._preprocess(A_bar=A_bar, B_bar=B_bar, C=self.C)

        # Iterate over time steps
        for t in range(L):
            h = self._recurrent_step(A_bar, B_bar, C, x, y, h, t)
        print(x.shape)

        return y + self.D * x

    def forward_convolutional(self, x):
        """
        Forward pass using the convolutional method.

        :param torch.Tensor x: The input tensor.
        :return: The output tensor.
        :rtype: torch.Tensor
        """

        # Store the sequence length
        L = x.shape[1]

        # Reshape input to [B, D, L]
        x_ = x.transpose(1, 2)

        # Compute the convolution kernel K
        K = self._compute_K(L)

        # Pad input and kernel to avoid circular convolution effects
        x_ = torch.nn.functional.pad(x_, (0, L))
        K = torch.nn.functional.pad(K, (0, L))

        # Compute FFT of input and kernel
        x_fft = torch.fft.rfft(x_, dim=2)
        K_fft = torch.fft.rfft(K, dim=1).unsqueeze(0)

        # Element-wise multiplication in frequency domain
        y_fft = x_fft * K_fft

        # Inverse FFT
        y = torch.fft.irfft(y_fft, n=2 * L, dim=2)

        return y[:, :, :L].transpose(1, 2) + self.D * x

    def change_forward(self, method):
        """
        Change the forward method.

        :param str method: The forward computation method. Available options
            are: recurrent, convolutional.
        :raises ValueError: If an invalid `method` is provided.
        """
        if method == "recurrent":
            self.forward = self.forward_recurrent
        elif method == "convolutional":
            self.forward = self.forward_convolutional
        else:
            raise ValueError(f"Unknown method: {method}")

    @abstractmethod
    def _compute_K(self, L):
        """
        Computation of the kernel K used in the convolutional method.

        :param int L: The length of the sequence.
        :return: The convolution kernel :math:`K`.
        :rtype: torch.Tensor
        """

    @staticmethod
    def _preprocess(A_bar, B_bar, C):
        """
        Preprocessing of the discretized matrices A_bar and B_bar.

        :return: The preprocessed matrices A_bar, B_bar, and C.
        :rtype: tuple
        """
        return A_bar, B_bar, C
