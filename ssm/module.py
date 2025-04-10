import torch
from lightning import LightningModule
from torchmetrics.classification import Accuracy


class SSMModule(LightningModule):
    """
    LightningModule for the SSM model.
    This module handles the training, validation, and testing steps.
    """

    def __init__(
        self,
        model,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={"lr": 1e-3},
    ):
        """
        Initialize the SSMModule by setting up the loss function and model.
        :param torch.nn.Module model: Model to be used.
        """
        super().__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.model = model
        self.accuracy = Accuracy(
            task="multiclass", num_classes=16, ignore_index=-1
        )
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs

    def forward(self, x):
        """
        Forward pass over the model.
        :param torch.Tensor x: Input tensor (batch_size, seq_len, input_dim).
        :return: Output tensor (batch_size, seq_len, output_dim).
        :rtype: Torch.Tensor
        """
        return self.model(x)

    def _iteration(self, batch, stage):
        """
        Perform a single iteration of training, validation, or testing.
        :param batch: Tuple containing input and target tensors.
        :type batch: tuple(torch.Tensor, torch.Tensor)
        :param str stage: Stage of the iteration (train, val, test).
        :return: Loss value.
        :rtype: torch.Tensor
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat.permute(0, 2, 1), y)
        self.log(
            f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log(
            f"{stage}_accuracy",
            self.accuracy(y_hat.permute(0, 2, 1), y),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def training_step(self, batch):
        """
        Perform a single training step.
        :param batch: Tuple containing input and target tensors.
        :type batch: tuple(torch.Tensor, torch.Tensor)
        :return: Loss value to be backpropagated.
        :rtype: torch.Tensor
        """
        return self._iteration(batch, "train")

    def validation_step(self, batch):
        """
        Perform a single validation step.

        :param batch: A tuple containing input and target tensors.
        :type batch: tuple(torch.Tensor, torch.Tensor)
        :return: Loss value.
        :rtype: torch.Tensor
        """
        return self._iteration(batch, "val")

    def test_step(self, batch):
        """
        Perform a single test step.
        :param batch: A tuple containing input and target tensors.
        :type batch: tuple(torch.Tensor, torch.Tensor)
        :return: Loss value.
        :rtype: torch.Tensor
        """
        return self._iteration(batch, "test")

    def configure_optimizers(self):
        """
        Configure the optimizer for the model.
        :return: Optimizer instance properly initialized with model parameters.
        :rtype: torch.optim.Optimizer
        """
        return self.optimizer_class(self.parameters(), **self.optimizer_kwargs)
