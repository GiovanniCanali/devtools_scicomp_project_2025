import torch
from torch.nn import LayerNorm
from torch.nn.modules.normalization import RMSNorm
from .block import S6Block
from .block.mixing_block import MixingBlock


class S6(torch.nn.Module):
    """
    Implementation of the Selective Structured State Space Sequence (S6) model.

    The S6 model is designed to efficiently model long sequences using selective
    state space models. Its selection mechanism allows it to focus on relevant
    parts of the input sequence, making it suitable for tasks such as selective
    copy. It enables improved scalability and performance compared to classical
    recurrent architectures. The parallel scan algorithm is used to compute the
    output efficiently.

    The model is composed of several S6 blocks, each followed by an activation
    function and a linear layer, similarly to the S4 model.

    .. seealso::
        **Original Reference**: Gu, A., Dao, T. (2024).
        "Mamba: Linear-Time Sequence Modeling with Selective State Spaces".
        arXiv:2312.00752.
        DOI: `<https://arxiv.org/abs/2312.00752>_`.

        **Original Reference**: Heinsen, F., A. (2023)
        "Efficient Parallelization of a Ubiquitous Sequential Computation".
        arXiv:2311.06281.
        DOI: `<https://arxiv.org/abs/2311.06281>_`.
    """

    def __init__(
        self,
        model_dim,
        hid_dim=16,
        n_layers=2,
        activation=torch.nn.GELU,
        real_random=False,
        normalization=True,
        **kwargs,
    ):
        """
        Initialization of the S6 model.

        :param int model_dim: The input dimension of the S6 block.
        :param int hid_dim: The hidden dimension of the S6 block.
        :param int n_layers: Number of S6 blocks. Default is 2.
        :param torch.nn.Module activation: The activation function.
            Default is `torch.nn.GELU`.
        :param bool real_random: If `True`, the real part of the A matrix of the
            diagonal block is initialized at random between 0 and 1.
            Default is `False`.
        :param bool normalization: If `True`, layer normalization is applied
            after each S6 block. Default is `True`.
        :param dict kwargs: Additional keyword arguments used in the block.
        """
        super().__init__()
        self.model_dim = model_dim

        # Initialize the layers
        layers = []
        for _ in range(n_layers):
            if normalization:
                layers.append(RMSNorm(model_dim, elementwise_affine=False))
            layers.append(
                S6Block(
                    model_dim=model_dim,
                    hid_dim=hid_dim,
                    real_random=real_random,
                    **kwargs,
                )
            )
            layers.append(activation())
            layers.append(MixingBlock(model_dim))
        self.layers = torch.nn.Sequential(*layers)
        self.norm_layer = (
            RMSNorm(model_dim, elementwise_affine=False)
            if normalization
            else None
        )

    def forward(self, x):
        """
        Forward pass of the S6 model.

        :param torch.Tensor x: The input tensor.
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        x = self.layers(x)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        return x
