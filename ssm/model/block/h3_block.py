import torch
from . import S4ShiftBlock, S4DBlock


class H3Block(torch.nn.Module):
    r"""
    Implementation of the H3 block.

    This block leverages the S4 shift block and the S4 diagonal block to create
    a hybrid architecture that mimics the behavior of the attention mechanism
    while maintaining the efficiency of state space models.

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
        heads,
        dt=0.1,
        initialization="S4D-Inv",
        discretization="bilinear",
        real_random=False,
        imag_random=False,
        **kwargs,
    ):
        """
        Initialization of the H3 block.

        :param int input_dim: The input dimension.
        :param int hid_dim: The hidden state dimension.
        :param str method: The forward computation method. Available options
            are: recurrent, convolutional.
        :param int heads: The number of attention heads. It must be a divisor of
            the input dimension.
        :param float dt: The time step for discretization. Default is `0.1`.
        :param str initialization: The method for initializing the A matrix in
            the diagonal S4 block. Options are: S4D-Inv, S4D-Lin, S4D-Quad,
            S4D-Real, real, complex. Default is `"S4D-Inv"`.
        :param str discretization: The method for discretizing the dynamics of
            the diagonal S4 block.
            Options are: bilinear, zoh. Default is `"bilinear"`.
        :param bool real_random: If `True`, the real part of the A matrix of the
            diagonal block is initialized at random between 0 and 1.
            Default is `False`.
        :param bool imag_random: If `True`, the imaginary part of the A matrix
            of the diagonal block is initialized at random between 0 and 1.
            Default is `False`.
        :param dict kwargs: Additional arguments for the class constructor.
        :raises ValueError: If the number of heads is not a divisor of the
            input dimension.
        """
        super().__init__()

        # Initialize number of heads
        if input_dim % heads != 0:
            raise ValueError(
                "The number of heads must be a divisor of the input dimension."
            )
        self.heads = heads
        self.dh = input_dim // heads

        # Initialize the shift S4 block
        self.shift_block = S4ShiftBlock(
            input_dim=input_dim,
            hid_dim=hid_dim,
            method=method,
            dt=dt,
            **kwargs,
        )

        # Initialize the diagonal S4 block
        self.diagonal_block = S4DBlock(
            input_dim=input_dim * self.dh,
            hid_dim=hid_dim,
            method=method,
            dt=dt,
            initialization=initialization,
            discretization=discretization,
            real_random=real_random,
            imag_random=imag_random,
            **kwargs,
        )

        # Initialize the weight matrixes
        self.linear_Q = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.linear_K = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.linear_V = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.linear_O = torch.nn.Linear(input_dim, input_dim, bias=False)

    def forward(self, x):
        """
        Forward pass of the H3 block.

        :param torch.Tensor x: The input tensor.
        :return: The output tensor after applying the H3 block.
        :rtype: torch.Tensor
        """
        # Save shapes
        batch_size, seq_len, _ = x.shape

        # Compute the Q, K, and V
        Q = self.linear_Q(x)
        K = self.linear_K(x)
        V = self.linear_V(x)

        # Pass K through the shift S4 block
        K = self.shift_block(K)

        # Reshape Q, K, and V for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.heads, self.dh).transpose(1, 2)
        K = K.reshape(batch_size, seq_len, self.heads, self.dh).transpose(1, 2)
        V = V.reshape(batch_size, seq_len, self.heads, self.dh).transpose(1, 2)

        # Compute batched outer product of K and V
        KV = K.unsqueeze(-1) * V.unsqueeze(-2)

        # Pass KV through the diagonal S4 block
        KV = KV.permute(0, 2, 1, 3, 4).reshape(batch_size, seq_len, -1)
        KV = (
            self.diagonal_block(KV)
            .reshape(batch_size, seq_len, self.heads, self.dh, self.dh)
            .permute(0, 2, 1, 3, 4)
        )

        # Compute O
        O = (
            torch.einsum("bhnc,bhncd->bhnd", Q, KV)
            .transpose(1, 2)
            .reshape(batch_size, seq_len, -1)
        )

        return self.linear_O(O)
