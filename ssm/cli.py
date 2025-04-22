import yaml
import importlib
import argparse
from copy import deepcopy
from .dataset import CopyDataset
from .trainer import Trainer
from .logger import Logger


class TrainingCLI:
    """
    Command Line Interface for training models.
    This class is responsible for loading the model and dataset
    from the configuration file, initializing the trainer,
    and starting the training process. It allows a straightforward
    way to configure and run training jobs from the command line, by simply
    providing a configuration file in YAML format.
    """

    def __init__(self, config_file=None):
        """
        Initialize the TrainingCLI class. It initializes the model and
        dataset, and sets up the trainer for training, based on the
        configuration file provided. If no configuration file is provided,
        it will parse command line arguments to get the configuration file.

        :param str config_file: Path to the configuration file.

        """
        self.args = self.argparsing()[0]
        if config_file is None:
            config_file = self.args.config_file
        config = self.load_config(config_file)
        model = self.init_model(deepcopy(config["model"]))
        dataset = CopyDataset(**deepcopy(config["dataset"]))
        logger = Logger(**deepcopy(config["logger"]))
        trainer_config = deepcopy(config["trainer"])
        trainer_config["dataset"] = dataset
        trainer_config["logger"] = logger
        self.trainer = self.init_trainer(trainer_config, model, dataset)
        self.write_on_tensorboard(config)

    def load_config(self, config_file):
        """
        Configure the training parameters.
        :param str config_file: Path to the configuration file.
        :return: Configuration dictionary.
        :rtype: dict
        """
        import os

        path = "/".join(config_file.split("/")[:-1]) + "/common.yaml"
        if os.path.exists(path):
            with open(path, "r") as f:
                common_config = yaml.safe_load(f)
        else:
            common_config = {}
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        # Combine common config with specific config
        if common_config:
            # Iterate over common config and add to specific config
            for k, v in common_config.items():
                # If section is not in specific config, add it
                if k not in config:
                    config[k] = v
                    continue
                # If section is in both, merge them
                if isinstance(v, dict):
                    # If section is a dict, merge it
                    for kk, vv in v.items():
                        # If key is not in specific config, add it
                        if kk not in config[k]:
                            config[k][kk] = vv
        return config

    @staticmethod
    def init_model(model_config):
        """
        Load the model from the configuration.
        :param dict model_config: Model configuration dictionary.
        :return: Model instance.
        :rtype: torch.nn.Module
        """
        for k, v in model_config.items():
            if isinstance(v, str) and (
                v.startswith("torch.") or v.startswith("ssm.")
            ):
                module_name, class_name = v.rsplit(".", 1)
                module = importlib.import_module(module_name)
                model_config[k] = getattr(module, class_name)
        return model_config.pop("model")(**model_config)

    @staticmethod
    def init_trainer(trainer_config, model, dataset):
        """
        Initialize the trainer with the given configuration.

        :param dict trainer_config: Trainer configuration dictionary.
        :param model: Model instance.
        :param dataset: Dataset instance.
        :return: Trainer instance.
        :rtype: Trainer
        """
        trainer_config["model"] = model
        trainer_config["dataset"] = dataset
        if "optimizer_class" in trainer_config:
            optimizer_class = trainer_config.pop("optimizer_class")
            module_name, class_name = optimizer_class.rsplit(".", 1)
            module = importlib.import_module(module_name)
            optimizer_class = getattr(module, class_name)
            trainer_config["optimizer_class"] = optimizer_class
        return Trainer(**trainer_config)

    @staticmethod
    def argparsing():
        """
        Parse command line arguments.
        :return: Parsed arguments.
        :rtype: argparse.Namespace
        """
        parser = argparse.ArgumentParser(description="Training CLI")
        parser.add_argument(
            "--config_file",
            type=str,
            help="Path to the configuration file",
        )
        parser.add_argument(
            "--fit",
            type=bool,
            default=True,
            help="Train the model",
        )
        parser.add_argument(
            "--test",
            type=bool,
            default=False,
            help="Test the model",
        )
        return parser.parse_known_args()

    def fit(self):
        """
        Start the training process.
        :return: None
        """
        self.trainer.fit()

    def test(self):
        """
        Start the testing process.
        :return: None
        """
        self.trainer.test()

    def __call__(self):
        """
        Call the train method to start training.
        :return: None
        """
        if self.args.fit:
            self.fit()
        if self.args.test:
            self.test()

    def write_on_tensorboard(self, config):
        """
        Write the configuration on TensorBoard.
        :param dict config: Configuration dictionary.
        """
        if hasattr(self.trainer, "writer"):
            config_str = yaml.dump(config)
            self.trainer.writer.add_text("config", config_str, global_step=0)
            self.trainer.writer.flush()

    def initialize_logger(self, config):
        """
        Initialize the logger for TensorBoard.
        :param dict config: Configuration dictionary.
        """
        logger = config.get("logger", {})
