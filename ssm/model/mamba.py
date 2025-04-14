import torch
from .block import MambaBlock


class Mamba(torch.nn.Module):
    def __init__(self, n_layers, output_dim=None, **kwargs):
        """
        Initializes the Mamba model, by building a stack of Mamba blocks.
        :param int n_layers: number of Mamba blocks in the stack.
        :param int output_dim: dimension of the output layer. If None, the
            output layer is an identity layer.
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
        self.input_dim = kwargs["input_dim"]
        self.mamba_blocks = torch.nn.Sequential(*mamba_blocks)
        if output_dim is not None:
            self.output_net = torch.nn.Linear(kwargs["input_dim"], output_dim)
        else:
            self.output_net = torch.nn.Identity()

    def forward(self, x):
        """
        Forward pass through the Mamba model. It iterates over the Mamba blocks
        and applies them sequentially to the input tensor.

        :param torch.Tensor x: Input tensor.
        :return: Output tensor.
        :rtype: torch.Tensor
        """
        x = self.mamba_blocks(x)
        return self.output_net(x)
