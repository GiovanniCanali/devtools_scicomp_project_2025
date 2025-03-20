import torch


def compute_hippo(N):
    """
    Definition of the HIPPO initialization for the hidden-to-hidden matrix.

    :param int N: The shape of the matrix.
    :return: A :math:`(N, N)` matrix initialized with the HIPPO method.
    :rtype: torch.Tensor
    """
    P = torch.sqrt(torch.arange(1, 2 * N, 2, dtype=torch.float32))
    A = 0.5 * (P[:, None] * P[None, :])
    A = torch.tril(A, diagonal=-1) - torch.diag(
        torch.arange(N, dtype=torch.float32)
    )
    return -A
