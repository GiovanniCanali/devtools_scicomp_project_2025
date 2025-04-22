import os
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    A logger class for training and evaluation.
    This class handles logging of training metrics, including loss and
    accuracy, to TensorBoard and the console. It also manages the
    initialization of the TensorBoard writer and the progress bar.
    """

    def __init__(
        self,
        logging_steps,
        logging_dir=None,
        enable_progress_bar=True,
        tensorboard_logger=True,
    ):
        """
        Initialize the Logger class.
        :param int logging_steps: The number of steps between logging.
        :param str logging_dir: The directory to save the logs.
        :param bool enable_progress_bar: Whether to enable the progress bar.
        :param bool tensorboard_logger: Whether to enable TensorBoard logging.
        """
        self.loss = 0
        self.accuracy = 0
        self.steps = 0
        self.enable_progress_bar = enable_progress_bar
        self.logging_steps = logging_steps
        self.logging_dir = self.logging_folder(logging_dir)
        os.makedirs(self.logging_dir, exist_ok=True)
        self.writer = (
            None if not tensorboard_logger else self.initialize_tensorboard()
        )
        self.pbar = None

    def initialize_tensorboard(self):
        """
        Initialize the TensorBoard writer.
        :return: A TensorBoard writer instance.
        :rtype: torch.utils.tensorboard.SummaryWriter
        """
        return SummaryWriter(log_dir=self.logging_dir)

    def initialize(self):
        """
        Initialize the logger.
        """
        self.loss = 0
        self.accuracy = 0

    def step(self, loss, accuracy):
        """
        Log a step.
        :param float loss: The loss value.
        :param float accuracy: The accuracy value.
        """
        self.loss += loss
        self.accuracy += accuracy
        self.steps += 1
        if self.steps % self.logging_steps == 0:
            log_loss = self.loss / self.logging_steps
            log_accuracy = self.accuracy / self.logging_steps
            self.log_on_tensorboard(
                "train/loss",
                log_loss,
                self.steps,
            )
            self.log_on_tensorboard(
                "train/accuracy",
                log_accuracy,
                self.steps,
            )
            self.writer.flush()
            self.pbar.set_postfix(
                loss=log_loss,
                accuracy=log_accuracy,
            )
            self.initialize()

    @staticmethod
    def logging_folder(base_dir):
        """
        Determine the next available logging folder based on existing
        directories in the specified base directory. The folder names are
        expected to follow the format "version_X", where X is an integer
        representing the version number.
        :param str base_dir: The base directory where the logging folders are
            located.
        :return: The path to the next available logging folder.
        :rtype: str
        """
        if base_dir is None:
            warnings.warn(
                "No logging directory provided. Using default directory."
            )
            base_dir = "logging_dir/default"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        idx = [
            name.split("version_")[-1]
            for name in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, name))
            and name.startswith("version_")
        ]
        idx = [int(i) for i in idx]
        if len(idx) == 0:
            return os.path.join(base_dir, "version_0")
        idx = max(idx) + 1
        folder = os.path.join(base_dir, f"version_{idx}")
        return folder

    def write_model_summary(
        self, trainable_parameters, non_trainable_parameters
    ):
        """
        Write the model summary to TensorBoard.
        :param torch.nn.Module model: The model to summarize.
        """
        if self.writer is not None:
            self.writer.add_text(
                "trainable_parameters", str(trainable_parameters), 0
            )
            self.writer.add_text(
                "non_trainable_parameters", str(non_trainable_parameters), 0
            )
            self.writer.flush()

    def log_on_tensorboard(self, name, value, step):
        """
        Write a scalar value to TensorBoard.
        :param str name: The name of the scalar value.
        :param float value: The value to write.
        :param int step: The current step number.
        """
        if self.writer is not None:
            self.writer.add_scalar(name, value, step)
            self.writer.flush()

    def initialize_pbar(self, steps):
        """
        Initialize the progress bar.
        :param int steps: The total number of steps.
        """
        self.pbar = tqdm(range(steps), disable=not self.enable_progress_bar)

    def save_model(self, model):
        """
        Save the model to a file.
        :param torch.nn.Module model: The model to save.
        :param str path: The path to save the model.
        """
        torch.save(
            model.state_dict(), os.path.join(self.logging_dir, "model.pth")
        )
