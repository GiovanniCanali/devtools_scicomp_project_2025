import torch
from .block import MambaBlock


class Mamba(torch.nn.Module):
    def __init__(self, n_layers, normalization=True, **kwargs):
        """
        Initializes the Mamba model, by building a stack of Mamba blocks.
        :param int n_layers: number of Mamba blocks in the stack.
        :param bool normalization: whether to apply layer normalization
        :param dict kwargs: arguments for the MambaBlock constructor.

        .. seealso::
            **Original Reference**: Gu, A. and Dao, T. (2024).
            "Mamba: Linear-Time Sequence Modeling with Selective State Spaces".
            arXiv:2312.00752.
            DOI: `<https://doi.org/10.48550/arXiv.2312.00752>_`.
        """
        super().__init__()
        self.n_layers = n_layers
        mamba_blocks = []

        for _ in range(n_layers):
            if normalization:
                mamba_blocks.append(torch.nn.RMSNorm(kwargs["model_dim"]))
            mamba_blocks.append(MambaBlock(**kwargs))

        self.model_dim = kwargs["model_dim"]
        self.mamba_blocks = torch.nn.Sequential(*mamba_blocks)
        self.norm_layer = (
            torch.nn.RMSNorm(kwargs["model_dim"]) if normalization else None
        )

    def forward(self, x):
        """
        Forward pass through the Mamba model. It iterates over the Mamba blocks
        and applies them sequentially to the input tensor.

        :param torch.Tensor x: Input tensor.
        :return: Output tensor.
        :rtype: torch.Tensor
        """
        x = self.mamba_blocks(x)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        return x
