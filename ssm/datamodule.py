from .dataset import CopyDataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader


class DataModule(LightningDataModule):
    def __init__(
        self,
        N,
        sequence_len,
        mem_tokens,
        alphabet_size,
        selective=True,
        batch_size=32,
        num_workers=4,
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
    ):
        """
        Initialize the data module.
        :param int N: Number to sample to generate. If `None`, an iterable datasets are created.
        :param int sequence_len: Length of the input sequence.
        :param int mem_tokens: Number of tokens to be copied.
        :param int alphabet_size: Alphabet size (number of unique tokens).
        :param bool selective: Whether to use selective copy, defaults to True
        :param int batch_size: Batch size for the dataloaders, defaults to 32
        :param int num_workers: Number of workers for the dataloaders, defaults to 4
        :param float train_size: Percentage of data to use for training,
            defaults to 0.8
        :param float val_size: Percentage of data to use for validation,
            defaults to 0.1
        :param float test_size: Percentage of data to use for testing,
            defaults to 0.1
        """
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
        """
        Create a dataset of the given size.
        :param float size: Percentage of data to use for the dataset.
        :return: An instance of the dataset class.
        :rtype: Union[StaticCopyDataset, IterableCopyDataset]
        """
        return CopyDataset(
            N=int(size * self.N) if self.N is not None else None,
            sequence_len=self.sequence_len,
            mem_tokens=self.mem_tokens,
            alphabet_size=self.alphabet_size,
            selective=self.selective,
        )

    def setup(self, stage=None):
        """
        Setup the data module by creating train, validation, and test datasets.
        :param str stage: Stage of the data module (fit, test), defaults to
            `None`. If `None`, all datasets are created.
        """
        # Assign train/val/test datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = self._create_dataset(self.train_size)
            self.val_dataset = self._create_dataset(self.val_size)
        if stage == "test" or stage is None:
            self.test_dataset = self._create_dataset(self.test_size)

    def train_dataloader(self):
        """
        Create the training dataloader.
        :return: The training dataloader.
        :rtype: torch.utils.data.DataLoader
        """
        return self.dataloader_class(
            self.train_dataset,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        """
        Create the validation dataloader.
        :return: The validation dataloader.
        :rtype: torch.utils.data.DataLoader
        """
        return self.dataloader_class(
            self.val_dataset,
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        """
        Create the test dataloader.
        :return: The test dataloader.
        :rtype: torch.utils.data.DataLoader
        """
        return self.dataloader_class(
            self.test_dataset,
            batch_size=self.batch_size,
        )
