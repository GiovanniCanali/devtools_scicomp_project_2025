import torch


class Swish(torch.nn.Module):
    """
    Implementation of the Swish activation function.

    .. seealso::
        **Original Reference**: Ramachandran, P. Zoph, B., and Le, Q. (2017).
        "Swish: A Self-Gated Activation Function".
        arXiv:1710.05941.
        DOI: `<https://doi.org/10.48550/arXiv.1710.05941>_`.
    """

    def __init__(self, beta=1.0):
        """
        Initialization of the Swish activation function.

        :param float beta: The parameter for the Swish function. Default is 1.0.
        :raises ValueError: If `beta` is not a positive number.
        """
        super().__init__()

        if beta <= 0:
            raise ValueError("Beta must be a positive number.")
        self.beta = beta

    def forward(self, x):
        """
        Forward pass of the Swish activation function.

        :param torch.Tensor x: The input tensor.
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        return x * torch.sigmoid(self.beta * x)


class GatedMLPBlock(torch.nn.Module):
    """
    Implementation of the Gated MLP Block.

    This block combines two linear layers, with one of them passed through a
    non-linear activation function.
    """

    def __init__(
        self,
        model_dim,
        hid_dim,
        activation,
        beta=1.0,
    ):
        """
        Initialization of the Gated MLP block.

        :param int model_dim: The input dimension.
        :param int hid_dim: The hidden dimension.
        :param str activation: The activation function to use. Available options
            are: `"silu"`, `"swish"`. Default is `"silu"`.
        :param float beta: The parameter for the Swish function. Default is 1.0.
        :raises ValueError: If an invalid `activation` is provided.
        """
        super().__init__()

        # Initialize the linear layers
        self.linear_layer1 = torch.nn.Linear(model_dim, hid_dim)
        self.linear_layer2 = torch.nn.Linear(model_dim, hid_dim)
        self.output_layer = torch.nn.Linear(hid_dim, model_dim)

        # Initialize the activation function
        if activation == "silu":
            self.activation = torch.nn.SiLU()
        elif activation == "swish":
            self.activation = Swish(beta=beta)
        else:
            raise ValueError(
                f"Invalid activation: {activation},"
                " available options are: 'silu', 'swish'."
            )

    def forward(self, x):
        """
        Forward pass of the Gated MLP block.

        :param torch.Tensor x: The input tensor.
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        # Compute the linear transformations
        output1 = self.linear_layer1(x)
        output2 = self.linear_layer2(x)

        # Apply the activation function to the second output
        activated_output2 = self.activation(output2)

        return self.output_layer(output1 * activated_output2)
