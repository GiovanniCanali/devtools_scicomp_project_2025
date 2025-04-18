import torch
from .s4_block_interface import S4BlockInterface
from ...utils import compute_hippo, initialize_dt


class S4BaseBlock(S4BlockInterface):
    r"""
    Implementation of the basic S4 block.

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

    def __init__(
        self,
        input_dim,
        hid_dim,
        method,
        dt_min=0.001,
        dt_max=0.01,
        hippo=False,
        **kwargs,
    ):
        """
        Initialization of the basic S4 block.

        :param int input_dim: The input dimension.
        :param int hid_dim: The hidden state dimension.
        :param str method: The forward computation method. Available options
            are: recurrent, convolutional.
        :param float dt_min: The minimum time step for discretization.
            Default is `0.001`.
        :param float dt_max: The maximum time step for discretization.
            Default is `0.01`.
        :param bool hippo: Whether to use the HIPPO matrix for initialization.
            Default is `False`.
        :param dict kwargs: Additional arguments for the class constructor.
        """
        # Initialize matrices A, B, and C
        if hippo:
            A = compute_hippo(hid_dim).repeat(input_dim, 1, 1)
        else:
            A = torch.rand(input_dim, hid_dim, hid_dim)
        B = torch.rand(input_dim, hid_dim, 1)
        C = torch.rand(input_dim, 1, hid_dim)

        # Initialize the time step dt
        dt = (
            initialize_dt(
                input_dim=input_dim,
                dt_min=dt_min,
                dt_max=dt_max,
                inverse_softplus=False,
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        super().__init__(
            input_dim=input_dim,
            hid_dim=hid_dim,
            dt=dt,
            A=A,
            B=B,
            C=C,
            method=method,
        )

        self.register_buffer(
            "I", torch.eye(hid_dim).unsqueeze(0).expand(input_dim, -1, -1)
        )

    def _compute_K(self, L):
        """
        Computation of the kernel K used in the convolutional method.
        K is defined as :math:`K = [C A^0 B, C A^1 B, ..., C A^{L-1} B]`.

        :param int L: The length of the sequence.
        :return: The convolution kernel :math:`K`.
        :rtype: torch.Tensor
        """
        A_bar, B_bar = self._discretize()
        # Create kernel tensor: [input_dim, L]
        K = torch.zeros(self.input_dim, L, device=self.A.device)

        # Initialize A^0 for all channels (identity matrices)
        A_pow = self.I.clone()

        for i in range(L):
            # Compute C·A^i·B for all channels. Shape: [input_dim, 1, 1]
            CAB = torch.bmm(torch.bmm(self.C, A_pow), B_bar)

            # Squeeze to get scalar value per channel
            K[:, i] = CAB.squeeze(-1).squeeze(-1)

            # Update A^i to A^(i+1)
            A_pow = torch.bmm(A_bar, A_pow)

        return K

    def _discretize(self):
        """
        Discretization of the continuous-time dynamics to obtain the matrices
        :math:`A_{bar}` and :math:`B_{bar}`.
        """
        matrix_1 = self.I + 0.5 * self.A * self.dt
        matrix_2 = (self.I - 0.5 * self.A * self.dt).inverse()
        A_bar = matrix_2 @ matrix_1
        B_bar = matrix_2 @ self.B * self.dt
        return A_bar, B_bar

    @staticmethod
    def _preprocess(A_bar, B_bar, C):
        """
        Preprocessing of the discretized matrices A_bar and B_bar.

        :return: The preprocessed matrices A_bar, B_bar, and C.
        :rtype: tuple
        """
        A_bar = A_bar.transpose(1, 2).unsqueeze(0)
        B_bar = B_bar.squeeze(-1).unsqueeze(0)
        C = C.transpose(1, 2).unsqueeze(0)

        return A_bar, B_bar, C

    @staticmethod
    def _recurrent_step(A_bar, B_bar, C, x, y, h, t):
        """
        Recurrent step computation.

        :param torch.Tensor A_bar: The discretized hidden-to-hidden matrix.
        :param torch.Tensor B_bar: The discretized input-to-hidden matrix.
        :param torch.Tensor C: The hidden-to-output matrix.
        :param torch.Tensor x: The input tensor.
        :param torch.Tensor y: The output tensor.
        :param torch.Tensor h: The hidden state tensor.
        :param int t: The current time step.
        :return: The updated hidden state.
        :rtype: torch.Tensor
        """
        # Compute Ah + Bx
        Ah = torch.matmul(h.unsqueeze(-2), A_bar).squeeze(-2)
        Bx = x[:, t, :].unsqueeze(-1) * B_bar
        h = Ah + Bx

        # Compute output
        y[:, t, :] = torch.matmul(h.unsqueeze(-2), C).squeeze()

        return h
