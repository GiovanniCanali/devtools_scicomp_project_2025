import torch
from .block import GatedMLPBlock


class GatedMLP(torch.nn.Module):
    """
    Implementation of the Gated MLP model.

    This model consists of a stack of Gated MLP blocks, each of which combines
    two linear layers with a non-linear activation function.
    """

    def __init__(
        self, model_dim, hid_dim, n_layers, activation, beta=1.0, **kwargs
    ):
        """
        Initialization of the Gated MLP model.

        :param int model_dim: The input dimension.
        :param int hid_dim: The hidden dimension.
        :param int n_layers: The number of Gated MLP blocks in the stack.
        :param str activation: The activation function to use. Available options
            are: `"silu"`, `"swish"`.
        :param float beta: The parameter for the Swish function. Default is 1.0.
        """
        super().__init__()

        gated_mlp_blocks = [
            GatedMLPBlock(
                model_dim=model_dim,
                hid_dim=hid_dim,
                activation=activation,
                beta=beta,
            )
            for _ in range(n_layers)
        ]
        self.gated_mlp_blocks = torch.nn.Sequential(*gated_mlp_blocks)

    def forward(self, x):
        """
        Forward pass of the Gated MLP model.

        :param torch.Tensor x: The input tensor.
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        return self.gated_mlp_blocks(x)
