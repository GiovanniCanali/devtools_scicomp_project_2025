import torch
from torch.func import vmap
from torch.utils.data import IterableDataset


class CopyDataset(IterableDataset):
    """
    A dataset composed of an infinite number of samples for copy/selective
    copy tasks. It generates samples on-the-fly.
    """

    def __init__(
        self,
        sequence_len,
        mem_tokens,
        vocab_size,
        selective=True,
        batch_size=64,
    ):
        """
        Initialize the dataset.

        :param int sequence_len: Length of the input sequence.
        :param int mem_tokens: Number of tokens to be copied.
        :param int vocab_size: Size of the alphabet (number of unique tokens).
        :param bool selective: whether to use selective copy, defaults to True
        :param int batch_size: Number of samples to generate in each batch,
            defaults to 64.
        """
        super().__init__()
        self.sequence_len = sequence_len
        self.mem_tokens = mem_tokens
        self.vocab_size = vocab_size
        self.selective = selective
        self.batch_size = batch_size

    @staticmethod
    def make_selective(t):
        """
        Randomly permute the input tensor in order to create a selective copy
        dataset.

        :param torch.Tensor t: Input tensor to be permuted.
        :return: Permuted tensor.
        :rtype: torch.Tensor
        """
        idx = torch.randperm(t.size(0))
        return t[idx].clone()

    def generate_data(self):
        """
        Generate a dataset of input-output pairs.
        The input is a sequence of tokens, and the output is a copy of the
        input sequence with a specific token indicating the start of the copy.

        :param int N: Number of samples to generate.
        :return: Tuple of input and output tensors.
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """

        N = self.batch_size
        tokens = torch.randint(
            low=1,
            high=self.vocab_size - 1,
            size=(
                N,
                self.mem_tokens,
            ),
        )
        if self.selective:
            inds = torch.stack(
                [
                    torch.randperm(self.sequence_len + self.mem_tokens)[
                        : self.mem_tokens
                    ]
                    for _ in range(N)
                ],
                0,
            )
            inds = inds.reshape(
                (
                    N,
                    self.mem_tokens,
                )
            )
            inds, _ = inds.sort()
        else:
            inds = torch.arange(self.mem_tokens).repeat(
                (
                    N,
                    1,
                )
            )
        zeros_x = torch.zeros(
            (
                N,
                self.mem_tokens + self.sequence_len,
            ),
            dtype=torch.long,
        )
        zeros_x.scatter_(-1, inds, tokens)
        markers = (self.vocab_size - 1) * torch.ones(
            (
                N,
                self.mem_tokens,
            ),
            dtype=torch.long,
        )

        x = torch.cat([zeros_x, markers], dim=-1)
        y = torch.cat([tokens], dim=-1)

        return x, y

    def __iter__(self):
        """
        Create an iterator for the dataset.
        :return: An iterator that generates batches of input-output pairs.
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        while True:
            x, y = self.generate_data()
            yield x, y
