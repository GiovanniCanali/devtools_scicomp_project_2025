import pytest
import shutil
from ssm import TrainingCLI


@pytest.mark.parametrize(
    "config_file",
    [
        "tests/experiments_test/cli_test.yaml",
        "tests/experiments_test/cli_test_no_tensorboard.yaml",
    ],
)
def test_contructor(config_file):

    TrainingCLI(config_file)
    shutil.rmtree("tests/logs")


@pytest.mark.parametrize(
    "config_file",
    [
        "tests/experiments_test/cli_test.yaml",
        "tests/experiments_test/cli_test_no_tensorboard.yaml",
    ],
)
def test_dataset(config_file):

    cli = TrainingCLI(config_file)
    dataset = cli.trainer.dataset
    x, y = next(dataset)

    assert x.shape == (16, 70)
    assert y.shape == (16, 10)
    assert not all(
        [True if i > 0 and i < 4 else False for i in x[:, :10].flatten()]
    )

    shutil.rmtree("tests/logs/testing_logs")


@pytest.mark.parametrize(
    "config_file",
    [
        "tests/experiments_test/cli_test.yaml",
        "tests/experiments_test/cli_test_no_tensorboard.yaml",
    ],
)
def test_model(config_file):

    cli = TrainingCLI(config_file)
    model = cli.trainer.model.model
    print(model)

    assert model.layers[0][1].hid_dim == 16
    assert model.layers[0][1].method == "convolutional"

    shutil.rmtree("tests/logs/testing_logs")


@pytest.mark.parametrize(
    "config_file",
    [
        "tests/experiments_test/cli_test.yaml",
        "tests/experiments_test/cli_test_no_tensorboard.yaml",
    ],
)
def test_trainer(config_file):

    cli = TrainingCLI(config_file)
    trainer = cli.trainer

    assert trainer.steps == 15
    assert trainer.test_steps == 10
    assert trainer.metric_tracker.logging_steps == 5

    shutil.rmtree("tests/logs/testing_logs")


@pytest.mark.parametrize(
    "config_file",
    [
        "tests/experiments_test/cli_test.yaml",
        "tests/experiments_test/cli_test_no_tensorboard.yaml",
    ],
)
def test_fit(config_file):

    cli = TrainingCLI(config_file)
    trainer = cli.trainer
    trainer.fit()

    shutil.rmtree("tests/logs/testing_logs")


@pytest.mark.parametrize(
    "config_file",
    [
        "tests/experiments_test/cli_test.yaml",
        "tests/experiments_test/cli_test_no_tensorboard.yaml",
    ],
)
def test_test(config_file):

    cli = TrainingCLI(config_file)
    trainer = cli.trainer
    trainer.fit()
    trainer.test()

    shutil.rmtree("tests/logs/testing_logs")
