import os
import warnings
import torch
from tqdm import tqdm
from torchmetrics import Accuracy
from .model.block.embedding_block import EmbeddingBlock


class Trainer:
    def __init__(
        self,
        model,
        dataset,
        steps,
        logger,
        accumulation_steps=1,
        device=None,
        test_steps=0,
        optimizer_class=torch.optim.Adam,
        optimizer_params={"lr": 1e-3},
        scheduler_class=None,
        scheduler_params=None,
    ):
        """
        Initialize the Trainer class.

        :param torch.nn.Module model: The model to be trained.
        :param ssm.dataset.CopyDataset dataset: The dataset to be used for
            training.
        :param int steps: The number of training steps.
        :param int accumulation_steps: The number of steps to accumulate
            gradients before updating the model parameters.
        :param int logging_steps: The number of steps between logging. For
            logging it is meant both the update of the progress bar and the
            TensorBoard logging. If you want to log only the progress bar
            set `tensotrboard_logger` to `False`.
        :param torch.device device: The device to use for training (CPU or GPU).
        :param int test_steps: The number of test steps.
        :param torch.optim.Optimizer optimizer_class: The optimizer class to use.
        :param dict optimizer_params: The parameters for the optimizer.
        """
        self.dataset = iter(dataset)
        n_classes = dataset.alphabet_size
        self.model = EmbeddingBlock(model, n_classes, dataset.mem_tokens)
        self.steps = steps
        self.logger = logger
        self.accumulation_steps = accumulation_steps
        self.test_steps = test_steps
        self.device = device if device else self.set_device()
        self.optimizer = optimizer_class(
            self.model.parameters(), **optimizer_params
        )
        self.loss = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=n_classes)
        self.mem_tokens = dataset.mem_tokens
        self.model_summary()

    def fit(self):
        """
        Train the model using gradient accumulation.
        """
        self.move_to_device()
        self.model.train()
        self.logger.initialize_pbar(self.steps)

        accumulation_counter = 0
        accumulated_loss = 0.0

        for i in self.logger.pbar:
            # Get new sample
            x, y = next(self.dataset)
            x, y = x.to(self.device), y.to(self.device)

            # Forward pass and loss computation
            loss, accuracy = self.compute_metrics(self.model(x), y)

            loss = (
                loss / self.accumulation_steps
            )  # Scale the loss for accumulation
            loss.backward()

            # Accumulate loss and increment the counter
            accumulated_loss += loss
            accumulation_counter += 1

            # Update the model parameters every accumulation_steps
            if accumulation_counter % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                # Log the metrics
                self.logger.step(accumulated_loss.item(), accuracy.item())
                accumulated_loss = 0.0
        self.logger.save_model(self.model)

    def test(self):
        """
        Test the model
        """
        self.move_to_device()
        self.model.eval()
        pbar = tqdm(
            range(self.test_steps), disable=not self.logger.enable_progress_bar
        )
        accuracy = 0
        loss = 0
        with torch.no_grad():
            for i in pbar:
                # Get a batch of data
                x, y = next(self.dataset)
                x, y = x.to(self.device), y.to(self.device)

                # Forward pass
                loss_, accuracy_ = self.compute_metrics(self.model(x), y)
                loss += loss_.item()
                accuracy += accuracy_.item()
        print(f"Test Loss: {loss / self.test_steps}")
        print(f"Test Accuracy: {accuracy / self.test_steps}")

        self.logger.log_on_tensorboard(
            "test/loss", loss / self.test_steps, self.test_steps
        )
        self.logger.log_on_tensorboard(
            "test/accuracy", accuracy / self.test_steps, self.test_steps
        )
        self.logger.writer.close()

    def compute_metrics(self, output, y):
        """
        Compute the loss and accuracy metrics.
        :param torch.Tensor output: The model output.
        :param torch.Tensor y: The ground truth labels.
        :return: The loss and accuracy values.
        :rtype: tuple
        """
        output = output.permute(0, 2, 1)[..., -self.mem_tokens :]
        loss = self.loss(output, y)
        accuracy = self.accuracy(output, y)
        return loss, accuracy

    def move_to_device(self):
        """
        Move the model and loss function to the specified device.
        """
        self.model.to(self.device)
        self.accuracy.to(self.device)

    @staticmethod
    def set_device():
        """
        Determine the device to use for training (CPU or GPU). This method
        checks for the availability of CUDA and Metal Performance Shaders
        (MPS) on macOS. If neither is available, it defaults to CPU.
        :return: The device to use for training.
        :rtype: torch.device
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _count_parameters(self):
        """
        Count the number of trainable and non-trainable parameters in the model.
        :return: The number of trainable parameters.
        :rtype: int
        """
        trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        non_trainable = sum(
            p.numel() for p in self.model.parameters() if not p.requires_grad
        )
        return {
            "trainable": trainable,
            "non_trainable": non_trainable,
            "total": trainable + non_trainable,
        }

    def model_summary(self):
        """
        Print a summary of the model, including the number of parameters and
        the architecture.
        """
        num_params = self._count_parameters()
        print(f"Trainable parameters: {num_params['trainable']}")
        print(f"Non-trainable parameters: {num_params['non_trainable']}")
        if self.logger is not None:
            self.logger.write_model_summary(
                num_params["trainable"], num_params["non_trainable"]
            )
