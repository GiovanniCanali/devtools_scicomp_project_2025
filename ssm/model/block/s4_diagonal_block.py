import torch
from .s4_block_interface import S4BlockInterface
from ...utils import (
    compute_S4DInv,
    compute_S4DLin,
    compute_S4DQuad,
    compute_S4DReal,
    initialize_dt,
)


class S4DBlock(S4BlockInterface):
    r"""
    Implementation of the diagonal S4 block.

    This block is a variant of the S4 block that uses a diagonal matrix for the
    hidden-to-hidden dynamics. It is designed to simplify both the logic and
    implementation of the S4 block while maintaining the same functionality.

    This block supports two forward pass methods: recurrent, and convolutional.

    - **Recurrent**: It applies discretized dynamics for sequential processing.
    - **Convolutional**: It uses the Fourier transform to compute convolutions.

    The block is defined by the following equations:

    .. math::
        \dot{h}(t) = Ah(t) + Bx(t),
        y(t) = Ch(t),

    where :math:`h(t)` is the hidden state, :math:`x(t)` is the input,
    :math:`y(t)` is the output, :math:`A`is a hidden-to-hidden diagonal matrix,
    :math:`B` is the input-to-hidden matrix, and :math:`C` is the
    hidden-to-output matrix.

    .. seealso::
        **Original Reference**: Gu, A., Gupta, A., Goel, K., and Re, G. (2022).
        "On the Parameterization and Initialization of Diagonal State Space
        Models".
        arXiv:2206.11893.
        DOI: `<https://arxiv.org/pdf/2206.11893>_`.
    """

    def __init__(
        self,
        model_dim,
        hid_dim,
        method,
        dt_max=0.1,
        dt_min=0.001,
        initialization="S4D-Inv",
        real_random=False,
        imag_random=False,
        discretization="bilinear",
        **kwargs,
    ):
        """
        Initialization of the diagonal S4 block.

        :param int model_dim: The input dimension.
        :param int hid_dim: The hidden state dimension.
        :param str method: The forward computation method. Available options
            are: recurrent, convolutional.
        :param float dt_max: The maximum time step for discretization.
            Default is `0.01`.
        :param float dt_min: The minimum time step for discretization.
            Default is `0.001`.
        :param str initialization: The method for initializing the A matrix.
            Options are: S4D-Inv, S4D-Lin, S4D-Quad, S4D-Real.
            Default is `"S4D-Inv"`.
        :param bool real_random: If `True`, the real part of the A matrix is
            initialized at random between 0 and 1. Default is `False`.
        :param bool imag_random: If `True`, the imaginary part of the A matrix
            is initialized at random between 0 and 1. Default is `False`.
        :param str discretization: The method for discretizing the dynamics.
            Options are: bilinear, zoh. Default is `"bilinear"`.
        :param dict kwargs: Additional arguments for the class constructor.
        """
        # Initialize matrices A, B, and C
        A = self.initialize_A(
            hid_dim,
            init_method=initialization,
            real_random=real_random,
            imag_random=imag_random,
        )
        A = A.unsqueeze(0).repeat(model_dim, 1)
        B = torch.rand(model_dim, hid_dim)
        C = torch.rand(model_dim, hid_dim)

        # Initialize the time step dt
        dt = initialize_dt(
            dim=model_dim,
            dt_max=dt_max,
            dt_min=dt_min,
            inverse_softplus=False,
        )

        super().__init__(
            model_dim=model_dim,
            hid_dim=hid_dim,
            dt=dt,
            A=A,
            B=B,
            C=C,
            method=method,
        )

        # Discretization of the dynamics
        if discretization == "bilinear":
            self._discretize = self._discretize_bilinear
        elif discretization == "zoh":
            self._discretize = self._discretize_zoh
        else:
            raise ValueError(f"Unknown discretization method: {discretization}")

    def _discretize_bilinear(self):
        """
        Discretization of the continuous-time dynamics to obtain the matrices
        :math:`A_{bar}` and :math:`B_{bar}`.
        """
        dt = self.dt.clamp(min=1e-7, max=1e6)
        tmp = 1 + self.A * dt.unsqueeze(-1) / 2
        tmp2 = 1 - self.A * dt.unsqueeze(-1) / 2
        A_bar = 1 / tmp2 * tmp
        B_bar = 1 / tmp2 * self.B * dt.unsqueeze(-1)
        return A_bar, B_bar

    def _discretize_zoh(self):
        """
        Discretization of the continuous-time dynamics to obtain the matrices
        :math:`A_{bar}` and :math:`B_{bar}`.
        """
        dt = self.dt
        tmp = self.A * dt.unsqueeze(-1)
        A_bar = torch.exp(tmp)
        B_bar = (A_bar - 1) * self.B * dt.unsqueeze(-1) / (tmp + 1e-6)
        return A_bar, B_bar

    @staticmethod
    def vandermonde_matrix(L, A_bar):
        """
        Compute the Vandermonde matrix for the diagonal S4 block.

        :param int L: The length of the sequence.
        :return: The Vandermonde matrix.
        :rtype: torch.Tensor
        """
        exponents = torch.arange(L, device=A_bar.device)
        return A_bar.unsqueeze(-1) ** exponents

    def _compute_K(self, L):
        """
        Computation of the kernel K used in the convolutional method.
        """
        A_bar, B_bar = self._discretize()
        # Compute the Vandermonde matrix
        V = self.vandermonde_matrix(L, A_bar)

        # Compute the kernel K using the Vandermonde matrix
        S = B_bar * self.C
        return torch.bmm(S.unsqueeze(1), V).squeeze(1).real

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
        # Compute hidden state
        x_t = x[:, t, :]
        h = A_bar.unsqueeze(0) * h + B_bar.unsqueeze(0) * x_t.unsqueeze(2)

        # Compute output
        y[:, t, :] = torch.sum(h * C.unsqueeze(0), dim=-1).real

        return h

    @staticmethod
    def initialize_A(
        hid_dim, init_method, real_random=False, imag_random=False
    ):
        """
        Initialization of the A matrix.

        :param int hid_dim: The hidden state dimension.
        :param str init_method: The method for initializing the A matrix.
            Options are: S4D-Inv, S4D-Lin, S4D-Quad, S4D-Real.
        :param bool real_random: If `True`, the real part of the A matrix is
            initialized at random between 0 and 1. Default is `False`.
        :param bool imag_random: If `True`, the imaginary part of the A matrix
            is initialized at random between 0 and 1. Default is `False`.
        :return: The initialized A matrix.
        :rtype: torch.Tensor
        :raises ValueError: If an unknown initialization method is provided.
        """

        if init_method == "S4D-Inv":
            return compute_S4DInv(
                hid_dim, real_random=real_random, imag_random=imag_random
            )

        elif init_method == "S4D-Lin":
            return compute_S4DLin(
                hid_dim, real_random=real_random, imag_random=imag_random
            )

        elif init_method == "S4D-Quad":
            return compute_S4DQuad(hid_dim, real_random=real_random)

        elif init_method == "S4D-Real":
            return compute_S4DReal(hid_dim, real_random=real_random)
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")
