import pytest
from ssm import TrainingCLI
from ssm import Trainer
from ssm.dataset import CopyDataset


def test_contructor():
    TrainingCLI("tests/experiments_test/cli_test.yaml")
    import shutil

    shutil.rmtree("tests/logs/testing_logs")


def test_dataset():
    cli = TrainingCLI("tests/experiments_test/cli_test.yaml")
    dataset = cli.trainer.dataset
    x, y = next(dataset)
    assert x.shape == (16, 60)
    assert y.shape == (16, 60)
    assert not all(  # Check that the task is a copy task (non selective)
        [True if i > 0 and i < 4 else False for i in x[:, :10].flatten()]
    )
    import shutil

    shutil.rmtree("tests/logs/testing_logs")


def test_model():
    cli = TrainingCLI("tests/experiments_test/cli_test.yaml")
    model = cli.trainer.model.model
    assert model.input_dim == 5
    assert model.layers[0].hid_dim == 16
    assert model.layers[0].method == "convolutional"
    import shutil

    shutil.rmtree("tests/logs/testing_logs")


def test_trainer():
    cli = TrainingCLI("tests/experiments_test/cli_test.yaml")
    trainer = cli.trainer
    assert trainer.steps == 15
    assert trainer.test_steps == 10
    assert trainer.logging_steps == 5
    import shutil

    shutil.rmtree("tests/logs/testing_logs")


def test_fit():
    cli = TrainingCLI("tests/experiments_test/cli_test.yaml")
    trainer = cli.trainer
    trainer.fit()
    import shutil

    shutil.rmtree("tests/logs/testing_logs")


def test_test():
    cli = TrainingCLI("tests/experiments_test/cli_test.yaml")
    trainer = cli.trainer
    trainer.fit()
    trainer.test()
    import shutil

    shutil.rmtree("tests/logs/testing_logs")
