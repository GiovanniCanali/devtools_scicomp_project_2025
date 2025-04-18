import torch


class ResidualBlock(torch.nn.Module):
    """
    Implementation of a residual block used in the S4 model.

    This block is designed to efficiently model long sequences using selective
    state space models. Its selection mechanism allows it to focus on relevant
    parts of the input sequence, making it suitable for tasks such as selective
    copy.

    The output is computed in an efficient manner by leveraging the parallel
    scan algorithm.
    """

    def __init__(self, model):
        """
        Initialization of the Residual Block.

        :param int input_dim: The input dimension.
        :param int hid_dim: The hidden dimension.
        """
        super().__init__()
        self.model = model

    def forward(self, x):
        """
        Forward pass of the Residual Block.

        :param torch.Tensor x: The input tensor.
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        # Compute the output using the model
        y = self.model(x)

        # Add the residual connection
        return x + y
