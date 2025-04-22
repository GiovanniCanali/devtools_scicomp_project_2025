import shutil
from ssm import TrainingCLI


def test_contructor():

    TrainingCLI("tests/experiments_test/cli_test.yaml")
    shutil.rmtree("tests/logs/testing_logs")


def test_dataset():

    cli = TrainingCLI("tests/experiments_test/cli_test.yaml")
    dataset = cli.trainer.dataset
    x, y = next(dataset)

    assert x.shape == (16, 70)
    assert y.shape == (16, 10)
    assert not all(
        [True if i > 0 and i < 4 else False for i in x[:, :10].flatten()]
    )

    shutil.rmtree("tests/logs/testing_logs")


def test_model():

    cli = TrainingCLI("tests/experiments_test/cli_test.yaml")
    model = cli.trainer.model.model

    assert model.input_dim == 5
    assert model.layers[0].model[0].hid_dim == 16
    assert model.layers[0].model[0].method == "convolutional"

    shutil.rmtree("tests/logs/testing_logs")


def test_trainer():

    cli = TrainingCLI("tests/experiments_test/cli_test.yaml")
    trainer = cli.trainer

    assert trainer.steps == 15
    assert trainer.test_steps == 10
    assert trainer.logger.logging_steps == 5

    shutil.rmtree("tests/logs/testing_logs")


def test_fit():

    cli = TrainingCLI("tests/experiments_test/cli_test.yaml")
    trainer = cli.trainer
    trainer.fit()

    shutil.rmtree("tests/logs/testing_logs")


def test_test():

    cli = TrainingCLI("tests/experiments_test/cli_test.yaml")
    trainer = cli.trainer
    trainer.fit()
    trainer.test()

    shutil.rmtree("tests/logs/testing_logs")
