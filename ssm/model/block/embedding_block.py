import torch
from torch.nn import Softmax


class EmbeddingBlock(torch.nn.Module):
    def __init__(self, model, embedding_dim):
        """
        Initialize the embedding block.

        :param torch.nn.Module model: The model to be used.
        :param int embedding_dim: The dimension of the embedding layer.
        """
        super().__init__()
        self.embedding = torch.nn.Embedding(embedding_dim, model.input_dim)
        self.model = model
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
        Forward pass of the embedding block.

        :param torch.Tensor x: The input tensor.
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        x = self.embedding(x)
        x = self.model(x)
        return self.softmax(x)
