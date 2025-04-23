import torch
from .block import S6Block
from .block.residual_block import ResidualBlock
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
        input_dim,
        hid_dim,
        output_dim,
        n_layers=2,
        activation=torch.nn.GELU,
        real_random=False,
        residual=True,
        layer_norm=True,
        **kwargs,
    ):
        """
        Initialization of the S6 model.

        :param int input_dim: The input dimension of the S6 block.
        :param int hid_dim: The hidden dimension of the S6 block.
        :param int output_dim: The output dimension.
        :param int n_layers: Number of S6 blocks. Default is 2.
        :param torch.nn.Module activation: The activation function.
            Default is `torch.nn.GELU`.
        :param bool real_random: If `True`, the real part of the A matrix of the
            diagonal block is initialized at random between 0 and 1.
            Default is `False`.
        :param bool residual: If `True`, a residual connection is added to the
            output of each S6 block. Default is `True`.
        :param bool layer_norm: If `True`, layer normalization is applied after
            each S6 block. Default is `True`.
        :param dict kwargs: Additional keyword arguments used in the block.
        """
        super().__init__()
        self.input_dim = input_dim

        # Initialize the layers
        layers = []
        for _ in range(n_layers):
            tmp = torch.nn.Sequential(
                S6Block(
                    input_dim=input_dim,
                    hid_dim=hid_dim,
                    real_random=real_random,
                    **kwargs,
                ),
                activation(),
                MixingBlock(input_dim),
                *([torch.nn.LayerNorm(input_dim)] if layer_norm else []),
            )
            layers.append(tmp if not residual else ResidualBlock(tmp))
        self.layers = torch.nn.Sequential(*layers)

        # Initialize the decoder to match the output dimension
        self.decoder = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Forward pass of the S6 model.

        :param torch.Tensor x: The input tensor.
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        y = self.layers(x)
        return self.decoder(y)
