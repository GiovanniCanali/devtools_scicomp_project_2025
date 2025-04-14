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
        alphabet_size,
        selective=True,
        marker=-1,
        batch_size=64,
    ):
        """
        Initialize the dataset.

        :param int sequence_len: Length of the input sequence.
        :param int mem_tokens: Number of tokens to be copied.
        :param int alphabet_size: Size of the alphabet (number of unique tokens).
        :param bool selective: whether to use selective copy, defaults to True
        :param int marker: Marker token to indicate the start of the copy,
            defaults to -1.
        :param int batch_size: Number of samples to generate in each batch,
            defaults to 64.
        """
        super().__init__()
        self.sequence_len = sequence_len
        self.mem_tokens = mem_tokens
        self.alphabet_size = alphabet_size
        self.selective = selective
        self.marker = marker
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
            low=1, high=self.alphabet_size - 1, size=(N, self.mem_tokens)
        )
        padding = torch.zeros(
            (N, self.sequence_len - self.mem_tokens), dtype=torch.long
        )
        x = torch.cat([tokens, padding], dim=-1)
        if self.selective:
            x = vmap(self.make_selective, randomness="different")(x)
        mask_x = torch.ones((N, self.mem_tokens)) * self.alphabet_size - 1
        x = torch.cat([x, mask_x], dim=-1).to(torch.int64)
        mask_y = torch.ones((N, self.sequence_len)) * self.marker
        y = torch.cat([mask_y, tokens], dim=-1).to(torch.int64)
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
