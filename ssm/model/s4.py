import torch
from torch.func import vmap
from .block.s4_base_block import S4BaseBlock


class S4(torch.nn.Module):
    """
    Implementation of the Structured State Space Sequence (S4) model.

    The S4 model is designed for efficiently modeling long-range dependencies in
    sequential data using structured state space representations. It enables
    improved scalability and performance compared to traditional recurrent
    architectures.

    This class supports three implementations of the underlying block:

    - **Continuous**: It uses the block's continuous-time dynamics.
    - **Recurrent**: It applies discretized dynamics for sequential processing.
    - **Fourier**: It leverages the Fourier transform to compute convolutions.

    .. seealso::
        **Original Reference**: Gu, A., Goel, K., and Re, G. (2021).
        "Efficiently Modeling Long Sequences with Structured State Spaces".
        arXiv:2111.00396.
        DOI: `<https://doi.org/10.48550/arXiv.2111.00396>_`.
    """

    def __init__(
        self, method, input_dim, output_dim, hidden_dim, hippo=True, fixed=False
    ):
        """
        Initialization of the S4 model.

        :param str method: The forward computation method.
            Available options are: `"continuous"`, `"recurrent"`, `"fourier"`.
        :param int input_dim: The input dimension.
        :param int output_dim: The output dimension.
        :param int hidden_dim: The hidden dimension.
        :param bool hippo: Whether to use the Hippocampus mechanism.
        :param bool fixed: Whether to use fixed weights.
        """
        super().__init__()

        self.block = S4BaseBlock(
            method=method,
            hidden_dim=hidden_dim,
            hippo=hippo,
            input_dim=input_dim,
        )
        self.mixing_fc = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Forward pass of the S4 model.

        :param torch.Tensor x: The input tensor with shape `(B, L, H)`.
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        y = self.block(x)
        y = self.mixing_fc(y)
        return y

    def change_forward(self, method):
        """
        Change the forward method of each block, depending on chosen `method`.

        :param str method: The forward computation method.
            Available options are: `"continuous"`, `"recurrent"`, `"fourier"`.
        """

        self.block.change_forward(method)
