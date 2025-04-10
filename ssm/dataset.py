import torch
from torch.utils.data import Dataset
from torch.func import vmap
from torch.utils.data import Dataset, IterableDataset


class BaseCopyDataset:
    """
    A base class for generating copy datasets.
    This class generates a dataset where the input is a sequence of tokens,
    and the output is a copy of the input sequence with a specific token
    indicating the start of the copy.
    """

    def __init__(
        self,
        sequence_len,
        mem_tokens,
        alphabet_size,
        selective=True,
        marker=-1,
    ):
        """
        Initialize the dataset.

        :param int sequence_len: Length of the input sequence.
        :param int mem_tokens: Number of tokens to be copied.
        :param int alphabet_size: Size of the alphabet (number of unique
            tokens).
        :param bool selective: If True, the tokens to be copied are selected
            randomly.
        :param int marker: The token used to indicate the start of the copy.
        """
        self.sequence_len = sequence_len
        self.mem_tokens = mem_tokens
        self.alphabet_size = alphabet_size
        self.selective = selective
        self.marker = marker

    @staticmethod
    def make_selective(t):
        """
        Randomly permute the input tensor in order to create a selective copy
        dataset.
        :param t: Input tensor to be permuted.
        :type t: torch.Tensor
        :return: Permuted tensor.
        :rtype: torch.Tensor
        """
        idx = torch.randperm(t.size(0))
        return t[idx].clone()

    def generate_data(self, N):
        """
        Generate a dataset of input-output pairs.
        The input is a sequence of tokens, and the output is a copy of the
        input sequence with a specific token indicating the start of the copy.

        :param int N: Number of samples to generate.
        :return: Tuple of input and output tensors.
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
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


class CopyDataset(Dataset):
    """
    Factory class for creating a copy dataset. If  `N` is `None`, it creates
    an iterable dataset. Otherwise, it creates a static dataset.
    """

    def __new__(cls, N, **kwargs):
        """
        Create a new instance of the dataset class.
        :param N: Number of samples to generate. If `None`, an iterable dataset
            is created.
        :return: An instance of the dataset class.
        :rtype: Union[StaticCopyDataset, IterableCopyDataset]
        """
        if N is None:
            return IterableCopyDataset(N, **kwargs)
        else:
            return StaticCopyDataset(N, **kwargs)


class StaticCopyDataset(BaseCopyDataset, Dataset):
    """
    A dataset composed of a fixed number of samples for copy/selective copy
    tasks.
    """

    def __init__(
        self,
        N,
        sequence_len,
        mem_tokens,
        alphabet_size,
        selective=True,
        marker=-1,
    ):
        """
        Initialize the dataset.
        :param int N: Number of samples to generate.
        :param int sequence_len: Length of the input sequence.
        :param int mem_tokens: Number of tokens to be copied.
        :param int alphabet_size: Size of the alphabet (number of unique tokens).
        :param bool selective: whether to use selective copy, defaults to True
        :param int marker: Marker token to indicate the start of the copy,
            defaults to -1.
        """

        super().__init__(
            sequence_len,
            mem_tokens,
            alphabet_size,
            selective,
            marker,
        )
        self.len = N
        self.x, self.y = self.generate_data(N)

    def __len__(self):
        """
        Return the number of samples in the dataset.
        :return: Number of samples.
        :rtype: int
        """
        return self.len

    def __getitem__(self, idx):
        """
        Given an index, return the corresponding input-output pair.
        :param int idx: Index of the sample to retrieve.
        :return: Tuple of input and output tensors.
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        return self.x[idx], self.y[idx]


class IterableCopyDataset(BaseCopyDataset, IterableDataset):
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
        :param int N: Number of samples to generate.
        :param int sequence_len: Length of the input sequence.
        :param int mem_tokens: Number of tokens to be copied.
        :param int alphabet_size: Size of the alphabet (number of unique tokens).
        :param bool selective: whether to use selective copy, defaults to True
        :param int marker: Marker token to indicate the start of the copy,
            defaults to -1.
        :param int batch_size: Number of samples to generate in each batch,
            defaults to 64.
        """
        super().__init__(
            sequence_len,
            mem_tokens,
            alphabet_size,
            selective,
            marker,
        )
        self.batch_size = batch_size

    def __iter__(self):
        """
        Create an iterator for the dataset.
        :return: An iterator that generates batches of input-output pairs.
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        return iter(self.generate_data(self.batch_size))
