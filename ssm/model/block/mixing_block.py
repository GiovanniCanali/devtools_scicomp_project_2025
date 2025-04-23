import torch


class MixingBlock(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.mixing_layer = torch.nn.Sequential(
            torch.nn.Conv1d(self.input_dim, 2 * self.input_dim, kernel_size=1),
            torch.nn.GLU(dim=-2),
        )

    def forward(self, x):
        """
        Forward pass of the mixing block.

        :param torch.Tensor x: The input tensor.
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        return self.mixing_layer(x.transpose(1, 2)).transpose(1, 2)
