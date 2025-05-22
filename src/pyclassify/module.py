from lightning import LightningModule
from torchmetrics.classification import Accuracy
from torch.nn.functional import cross_entropy
import torch
from .model import AlexNet

class Classifier(LightningModule):
    def __init__(self):
       super().__init__()
       self.model = AlexNet(num_classes=10)
       self.train_accuracy = Accuracy(task="multiclass", num_classes=10)
       self.val_accuracy = Accuracy(task="multiclass", num_classes=10)
       self.test_accuracy = Accuracy(task="multiclass", num_classes=10)

    def _classifier_step(self, batch):
        features, true_labels = batch
        logits = self.model(features)
        loss = cross_entropy(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)
        return predicted_labels, true_labels, loss

    def training_step(self, batch, _):
        predicted_label, true_labels, loss = self._classifier_step(batch)
        self.log('train_loss', loss)
        self.log('train_accuracy',
                 self.train_accuracy(predicted_label, true_labels),
                 on_step=True,
                 on_epoch=False)
        return loss

    def validation_step(self, batch, _):
        predicted_label, true_labels, loss = self._classifier_step(batch)
        self.log('train_loss', loss)
        self.log('train_accuracy',
                 self.val_accuracy(predicted_label, true_labels),
                 on_step=True,
                 on_epoch=False)

    def test_step(self, batch, _):
        self.log('train_loss', loss)
        predicted_label, true_labels, loss = self._classifier_step(batch)
        self.log('train_accuracy',
                 self.test_accuracy(predicted_label, true_labels),
                 on_step=True,
                 on_epoch=False)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer
