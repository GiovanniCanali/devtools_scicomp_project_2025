import torch
from torch.utils.data import Dataset
from torch.func import vmap
from torch.utils.data import Dataset, IterableDataset


class BaseCopyDataset:
    def __init__(
        self,
        N,
        sequence_len,
        mem_tokens,
        alphabet_size,
        selective=True,
        marker=-1,
    ):
        self.sequence_len = sequence_len
        self.mem_tokens = mem_tokens
        self.alphabet_size = alphabet_size
        self.selective = selective
        self.marker = marker

    @staticmethod
    def make_selective(t):
        idx = torch.randperm(t.size(0))
        return t[idx].clone()

    def generate_data(self, N):
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

    def __new__(cls, N, **kwargs):
        if N is None:
            return IterableCopyDataset(N, **kwargs)
        else:
            return StaticCopyDataset(N, **kwargs)


class StaticCopyDataset(BaseCopyDataset, Dataset):

    def __init__(
        self,
        N,
        sequence_len,
        mem_tokens,
        alphabet_size,
        selective=True,
        marker=-1,
    ):
        super().__init__(
            N,
            sequence_len,
            mem_tokens,
            alphabet_size,
            selective,
            marker,
        )
        self.len = N
        self.x, self.y = self.generate_data(N)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class IterableCopyDataset(BaseCopyDataset, IterableDataset):
    """
    A dataset that returns a copy of the input data.
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
        super().__init__(
            N,
            sequence_len,
            mem_tokens,
            alphabet_size,
            selective,
            marker,
        )

    def __iter__(self):
        return iter(self.generate_data(64))
