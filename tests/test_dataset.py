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
    assert data[0].shape == (32, sequence_len + mem_tokens)
    assert data[1].shape == (32, sequence_len + mem_tokens)
    assert torch.isclose(
        data[1][:, :sequence_len],
        torch.ones((32, sequence_len), dtype=torch.int64) * -1,
    ).all()
    assert [
        True if i > 0 and i < 4 else False for i in data[0].flatten()
    ].count(True) == 32 * mem_tokens
    assert (
        not all(
            [
                True if i > 0 and i < 4 else False
                for i in data[0][:, :mem_tokens].flatten()
            ]
        )
        == selective
    )
