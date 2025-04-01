import torch
from lightning import LightningModule
from torchmetrics.classification import Accuracy


class SSMModule(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.model = model
        self.accuracy = Accuracy(
            task="multiclass", num_classes=16, ignore_index=-1
        )

    def forward(self, x):
        return self.model(x)

    def _iteration(self, batch, stage):
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
        return self._iteration(batch, "train")

    def validation_step(self, batch):
        return self._iteration(batch, "val")

    def test_step(self, batch):
        return self._iteration(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
