from ssm.dataset import CopyDataset


def test_constructor():
    CopyDataset(
        sequence_len=10,
        mem_tokens=5,
        alphabet_size=10,
        N=20,
        selective=True,
    )


def test_getitem():
    dataset = CopyDataset(
        sequence_len=10,
        mem_tokens=5,
        alphabet_size=10,
        N=10,
        selective=True,
    )
    x, y = dataset[0]
    assert x.shape == (15,)
    assert y.shape == (15,)
    x, y = dataset[:5]
    assert x.shape == (5, 15)
    assert y.shape == (5, 15)
