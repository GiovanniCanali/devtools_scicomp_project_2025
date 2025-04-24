import torch
from torch.nn import Softmax


class EmbeddingBlock(torch.nn.Module):
    def __init__(self, model, vocab_size, model_dim, mem_tokens, out_dim=None):
        """
        Initialize the embedding block.

        :param torch.nn.Module model: The model to be used.
        :param int embedding_dim: The dimension of the embedding layer.
        """
        super().__init__()
        self.embedding = torch.nn.Embedding(
            vocab_size, model_dim, padding_idx=0
        )
        self.model = model
        if out_dim is None:
            out_dim = vocab_size
        self.mem_tokens = mem_tokens
        self.project = torch.nn.Linear(model_dim, out_dim, bias=False)

    def forward(self, x):
        """
        Forward pass of the embedding block.

        :param torch.Tensor x: The input tensor.
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        x = self.embedding(x)
        x = self.model(x)[:, -self.mem_tokens :, :]
        return self.project(x)
