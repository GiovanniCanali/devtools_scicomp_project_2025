import pytest
import torch
from ssm import CopyDataset


def test_constructor():

    CopyDataset(
        sequence_len=20,
        batch_size=32,
        alphabet_size=10,
        mem_tokens=10,
        marker=-1,
        selective=False,
    )


@pytest.mark.parametrize("selective", [True, False])
@pytest.mark.parametrize("mem_tokens", [5, 7])
@pytest.mark.parametrize("sequence_len", [10, 20])
def test_generate_data(selective, mem_tokens, sequence_len):

    dataset = CopyDataset(
        sequence_len=sequence_len,
        batch_size=32,
        alphabet_size=5,
        mem_tokens=mem_tokens,
        marker=-1,
        selective=selective,
    )

    data = dataset.generate_data()
    assert data is not None
    assert len(data) == 2

    input_, target = data
    expected_shape = (32, sequence_len + mem_tokens)
    assert input_.shape == expected_shape
    assert target.shape == expected_shape

    expected_target = torch.ones((32, sequence_len), dtype=torch.int64) * -1
    assert torch.allclose(target[:, :sequence_len], expected_target)

    mask = (input_ > 0) & (input_ < 4)
    assert mask.sum() == 32 * mem_tokens

    mask = (input_[:, :mem_tokens] > 0) & (input_[:, :mem_tokens] < 4)
    assert not (mask.all().item() == selective)
