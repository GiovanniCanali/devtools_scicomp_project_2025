import torch
from .block import S4BaseBlock, S4LowRankBlock, S4DBlock, S6Block


class S4(torch.nn.Module):
    """
    Implementation of the Structured State Space Sequence (S4) model.

    The S4 model is designed for efficiently modeling long-range dependencies in
    sequential data using structured state space representations. It enables
    improved scalability and performance compared to traditional recurrent
    architectures.

    This class supports two implementations of the underlying block:

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
        self,
        input_dim,
        model_dim,
        hid_dim,
        output_dim,
        block_type="S4",
        n_layers=2,
        func=torch.nn.ReLU,
        **kwargs,
    ):
        """
        Initialization of the S4 model.

        :param int input_dim: The input dimension.
        :param int model_dim: The dimension of data passed S4 blocks
        :param int hidden_dim: The hidden dimension.
        :param int output_dim: The output dimension.
        :param int n_layers: Number of S4 layers.
        :param str block_type: The type of S4 block to use.
            Available options are: `"S4"`, `"S4D"`, `"S4LowRank"`.
        :param torch.nn.Module func: The activation function.
        :param dict kwargs: Additional keyword arguments used in the block.
        """
        super().__init__()
        if block_type == "S4":
            block_class = S4BaseBlock
        elif block_type == "S4D":
            block_class = S4DBlock
        elif block_type == "S4LowRank":
            block_class = S4LowRankBlock
        elif block_type == "S6":
            block_class = S6Block
        else:
            raise RuntimeError("Unrecognized method {method}")

        self.encoder = torch.nn.Linear(input_dim, model_dim)
        layers = []
        for _ in range(n_layers):
            layers.append(
                block_class(hid_dim=hid_dim, input_dim=model_dim, **kwargs)
            )
            layers.append(func())
            layers.append(torch.nn.Linear(model_dim, model_dim))
        self.layers = torch.nn.Sequential(*layers)
        self.decode = torch.nn.Linear(model_dim, output_dim)
        self.soft_max = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Forward pass of the S4 model.

        :param torch.Tensor x: The input tensor with shape `(B, L, H)`.
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        y = self.layers(x)
        y = self.decode(y)
        return y

    def change_forward(self, method):
        """
        Change the forward method of each block, depending on chosen `method`.

        :param str method: The forward computation method.
            Available options are: `"continuous"`, `"recurrent"`, `"fourier"`.
        """

        for layer in self.layers:
            if isinstance(layer, (S4BaseBlock, S4DBlock)):
                layer.change_forward(method)
