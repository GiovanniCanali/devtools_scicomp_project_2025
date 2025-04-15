import yaml
import importlib
import argparse
from .dataset import CopyDataset
from .trainer import Trainer


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
        self.args = self.argparsing()
        if config_file is None:
            config_file = self.args.config_file
        config = self.load_config(config_file)
        model = self.init_model(config["model"])
        dataset = CopyDataset(**config["dataset"])
        trainer_config = config["trainer"]
        trainer_config["dataset"] = dataset
        self.trainer = self.init_trainer(trainer_config, model, dataset)

    def load_config(self, config_file):
        """
        Configure the training parameters.
        :param str config_file: Path to the configuration file.
        :return: Configuration dictionary.
        :rtype: dict
        """
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
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
