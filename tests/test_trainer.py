import pytest
import shutil
import torch
from ssm import Trainer
from ssm.model import S4, S6, Mamba
from ssm.dataset import CopyDataset

input_dim = 10
hid_dim = 12
output_dim = 5
model_dim = 64

S4_model = S4(
    block_type="S4",
    method="convolutional",
    input_dim=input_dim,
    hid_dim=hid_dim,
    model_dim=model_dim,
    output_dim=output_dim,
    n_layers=2,
    hippo=True,
)

S6_model = S6(
    input_dim=input_dim,
    hid_dim=hid_dim,
    model_dim=model_dim,
    output_dim=output_dim,
    n_layers=2,
)

Mamba_model = Mamba(
    input_dim=input_dim,
    model_dim=model_dim,
    output_dim=output_dim,
    n_layers=1,
    expansion_factor=2,
    kernel_size=3,
    method="convolutional",
    hippo=True,
)

dataset = CopyDataset(
    sequence_len=50,
    batch_size=12,
    alphabet_size=5,
    mem_tokens=10,
    marker=-1,
    selective=False,
)


@pytest.mark.parametrize("model", [S4_model, S6_model, Mamba_model])
def test_constructor(model):

    Trainer(
        model=model,
        steps=100,
        test_steps=10,
        logging_steps=10,
        dataset=dataset,
        logging_dir="tests/logs/",
        device="cpu",
    )


@pytest.mark.parametrize("model", [S4_model, Mamba_model])
def test_fit(model):

    trainer = Trainer(
        model=model,
        dataset=dataset,
        steps=2,
        test_steps=10,
        logging_steps=10,
        logging_dir="tests/logs/",
        device="cpu",
    )

    trainer.fit()
    shutil.rmtree("tests/logs/")


@pytest.mark.parametrize("tensorboard_logger", [True, False])
def test_tensorboard_logging(tensorboard_logger):

    trainer = Trainer(
        model=S4_model,
        dataset=dataset,
        steps=2,
        test_steps=10,
        logging_steps=10,
        tensorboard_logger=tensorboard_logger,
        logging_dir="tests/logs/",
        device="cpu",
    )

    trainer.fit()
    if not tensorboard_logger:
        with pytest.raises(FileNotFoundError):
            shutil.rmtree("tests/logs/")
    else:
        shutil.rmtree("tests/logs/")


def test_custom_optimizer():

    trainer = Trainer(
        model=S4_model,
        dataset=dataset,
        steps=2,
        test_steps=10,
        logging_steps=10,
        optimizer_params={"lr": 0.05, "weight_decay": 0.1},
        logging_dir="tests/logs/",
        device="cpu",
    )

    assert isinstance(trainer.optimizer, torch.optim.Adam)
    assert trainer.optimizer.param_groups[0]["lr"] == 0.05
    assert trainer.optimizer.param_groups[0]["weight_decay"] == 0.1

    trainer = Trainer(
        model=S4_model,
        dataset=dataset,
        steps=2,
        test_steps=10,
        logging_steps=10,
        optimizer_class=torch.optim.SGD,
        optimizer_params={"lr": 0.07, "momentum": 0.9},
        logging_dir="tests/logs/",
        device="cpu",
    )

    assert isinstance(trainer.optimizer, torch.optim.SGD)
    assert trainer.optimizer.param_groups[0]["lr"] == 0.07
    assert trainer.optimizer.param_groups[0]["momentum"] == 0.9


@pytest.mark.parametrize("model", [S4_model, Mamba_model])
def test_test(model):

    trainer = Trainer(
        model=model,
        dataset=dataset,
        steps=2,
        test_steps=10,
        logging_steps=10,
        logging_dir="tests/logs/",
        device="cpu",
    )

    trainer.fit()
    trainer.test()
    shutil.rmtree("tests/logs/")


def test_accumulation_steps():

    trainer = Trainer(
        model=S4_model,
        dataset=dataset,
        steps=2,
        test_steps=10,
        logging_steps=10,
        accumulation_steps=2,
        logging_dir="tests/logs/",
        device="cpu",
    )

    trainer.fit()
    shutil.rmtree("tests/logs/")
