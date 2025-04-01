from .dataset import CopyDataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader


class DataModule(LightningDataModule):
    def __init__(
        self,
        N,
        sequence_len: int,
        mem_tokens: int,
        alphabet_size: int,
        selective: bool = True,
        batch_size: int = 32,
        num_workers: int = 4,
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
    ):
        super().__init__()
        self.N = N
        self.dataloader_class = DataLoader
        self.sequence_len = sequence_len
        self.mem_tokens = mem_tokens
        self.alphabet_size = alphabet_size
        self.selective = selective
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

    def _create_dataset(self, size):
        return CopyDataset(
            N=int(size * self.N) if self.N is not None else None,
            sequence_len=self.sequence_len,
            mem_tokens=self.mem_tokens,
            alphabet_size=self.alphabet_size,
            selective=self.selective,
        )

    def setup(self, stage=None):
        # Assign train/val/test datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = self._create_dataset(self.train_size)
            self.val_dataset = self._create_dataset(self.val_size)
        if stage == "test" or stage is None:
            self.test_dataset = self._create_dataset(self.test_size)

    def train_dataloader(self):
        return self.dataloader_class(
            self.train_dataset,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return self.dataloader_class(
            self.val_dataset,
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        return self.dataloader_class(
            self.test_dataset,
            batch_size=self.batch_size,
        )
