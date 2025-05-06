import torch
from torch.nn import LayerNorm
from torch.nn.modules.normalization import RMSNorm
from .block import S4BaseBlock, S4LowRankBlock, S4DBlock
from .block.mixing_block import MixingBlock


class S4(torch.nn.Module):
    """
    Implementation of the Structured State Space Sequence (S4) model.

    The S4 model is designed for efficiently modeling long-range dependencies in
    sequential data using structured state space representations.
    It enables improved scalability and performance compared to traditional
    recurrent architectures.

    The model is composed of several S4 blocks, each followed by an activation
    function and a linear layer, as explained in the referenced paper.
    The S4 blocks can be of different types, including the basic S4, and the
    low-rank and diagonal variants.

    Each block supports two forward pass methods:

    - **Recurrent**: It applies discretized dynamics for sequential processing.
    - **Convolutional**: It uses the Fourier transform to compute convolutions.

    .. warning::
        The low-rank S4 block supports only the convolutional forward pass.

    .. seealso::
        **Original Reference**: Gu, A., Goel, K., and Re, G. (2021).
        "Efficiently Modeling Long Sequences with Structured State Spaces".
        arXiv:2111.00396.
        DOI: `<https://doi.org/10.48550/arXiv.2111.00396>_`.
    """

    def __init__(
        self,
        model_dim,
        hid_dim,
        method,
        n_layers=2,
        block_type="S4",
        activation=torch.nn.GELU,
        normalization=True,
        **kwargs,
    ):
        """
        Initialization of the S4 model.

        :param int model_dim: The input dimension of the S4 block.
        :param int hid_dim: The hidden dimension of the S4 block.
        :param str method: The forward computation method for each S4 block.
            Available options are: `"recurrent"`, `"convolutional"`.
        :param int n_layers: Number of S4 blocks. Default is 2.
        :param str block_type: The type of S4 block to use. Available options
            are: `"S4"`, `"S4D"`, `"S4LowRank"`. Default is `"S4"`.
        :param torch.nn.Module activation: The activation function.
            Default is `torch.nn.ReLU`.
        :param bool normalization: If `True`, layer normalization is applied
            after each S4 block. Default is `True`.
        :param dict kwargs: Additional keyword arguments used in the block.
        :raises ValueError: If the specified `block_type` is not valid.
        """
        super().__init__()
        self.model_dim = model_dim
        # Initialize parameters
        self.block_type = block_type

        # Initialize the block class based on the specified type
        if self.block_type == "S4":
            block_class = S4BaseBlock
        elif self.block_type == "S4D":
            block_class = S4DBlock
        elif self.block_type == "S4LowRank":
            block_class = S4LowRankBlock
        else:
            raise ValueError(
                f"Invalid block type: {self.block_type}"
                "Available options are: 'S4', 'S4D', 'S4LowRank'."
            )

        layers = []
        for _ in range(n_layers):
            if normalization:
                layers.append(RMSNorm(model_dim, elementwise_affine=False))
            layers.append(
                block_class(
                    model_dim=model_dim,
                    hid_dim=hid_dim,
                    method=method,
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
        Forward pass of the S4 model.

        :param torch.Tensor x: The input tensor.
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        x = self.layers(x)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        return x

    def change_forward(self, method):
        """
        Change the forward method of each block, depending on chosen `method`.

        :param str method: The forward computation method.
            Available options are: `"recurrent"`, `"convolutional"`.
        """
        for layer in self.layers:
            if isinstance(layer, (S4BaseBlock, S4DBlock)):
                layer.change_forward(method)
