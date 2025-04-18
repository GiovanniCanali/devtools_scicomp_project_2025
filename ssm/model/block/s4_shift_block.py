import torch
from . import S4BaseBlock


class S4ShiftBlock(S4BaseBlock):
    r"""
    Implementation of the S4 block with shift dynamics.

    This block is a variant of the S4 block that uses a shift matrix for the
    hidden-to-hidden dynamics to create a memory of the previous state.
    In particular, matrix A is not trainable.

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
        **Original Reference**: Fu, D., Dao, T., et al. (2023).
        "Hungry Hungry Hippos: Towards Language Modeling with State Space
        Models".
        arXiv:2212.14052.
        DOI: `<https://doi.org/10.48550/arXiv.2212.14052>_`.
    """

    def __init__(
        self,
        input_dim,
        hid_dim,
        method,
        dt_min=0.001,
        dt_max=0.01,
        **kwargs,
    ):
        """
        Initialization of the S4 shift block.

        :param int input_dim: The input dimension.
        :param int hid_dim: The hidden state dimension.
        :param str method: The forward computation method. Available options
            are: recurrent, convolutional.
        :param float dt_min: The minimum time step for discretization.
            Default is `0.001`.
        :param float dt_max: The maximum time step for discretization.
            Default is `0.01`.
        :param dict kwargs: Additional arguments for the class constructor.
        """

        super().__init__(
            input_dim=input_dim,
            hid_dim=hid_dim,
            method=method,
            hippo=False,
            dt_min=dt_min,
            dt_max=dt_max,
            **kwargs,
        )

        # Remove the initialized A matrix
        del self.A

        # Initialize the shift matrix A as a non-trainable parameter
        A = self.initialize_A(hid_dim)
        A = A.repeat(input_dim, 1, 1)
        self.register_buffer("A", A)

    def initialize_A(self, hid_dim):
        """
        Initialize the shift matrix A.

        :param int hid_dim: The hidden state dimension.
        :return: The initialized shift matrix A.
        :rtype: torch.Tensor
        """
        A = torch.eye(hid_dim - 1, hid_dim)
        return torch.nn.functional.pad(A, (0, 0, 1, 0))
