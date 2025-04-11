import torch
from .block import MambaBlock


class Mamba(torch.nn.Module):
    def __init__(self, n_layers, **kwargs):
        """
        Initializes the Mamba model, by building a stack of Mamba blocks.
        :param int n_layers: number of Mamba blocks in the stack.
        :param dict kwargs: arguments for the MambaBlock constructor.

        .. seealso::
            **Original Reference**: Gu, A. and Dao, T. (2024).
            "Mamba: Linear-Time Sequence Modeling with Selective State Spaces".
            arXiv:2312.00752.
            DOI: `<https://doi.org/10.48550/arXiv.2312.00752>_`.
        """
        super().__init__()
        self.n_layers = n_layers
        mamba_blocks = torch.nn.ModuleList(
            [MambaBlock(**kwargs) for _ in range(n_layers)]
        )
        self.mamba_blocks = torch.nn.Sequential(*mamba_blocks)

    def forward(self, x):
        """
        Forward pass through the Mamba model. It iterates over the Mamba blocks
        and applies them sequentially to the input tensor.

        :param torch.Tensor x: Input tensor.
        :return: Output tensor.
        :rtype: torch.Tensor
        """
        return self.mamba_blocks(x)
