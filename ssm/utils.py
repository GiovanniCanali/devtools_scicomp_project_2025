import torch


def compute_hippo(N):
    """
    Constructs the HIPPO hidden-to-hidden matrix A.

    :param int N: The size of the HIPPO matrix.
    :return: A (N, N) matrix initialized using the HIPPO method.
    :rtype: torch.Tensor
    """
    P = torch.sqrt(torch.arange(1, 2 * N, 2))
    A = 0.5 * torch.outer(P, P)
    A = torch.tril(A, diagonal=-1)
    diag_indices = torch.arange(N) + 1
    A = A - torch.diag(diag_indices)
    return -A


def compute_hippo(N):
    """
    Constructs the HIPPO hidden-to-hidden matrix A.

    :param int N: The size of the HIPPO matrix.
    :return: A (N, N) matrix initialized using the HIPPO method
    :rtype: torch.Tensor
    """
    P = torch.sqrt(
        torch.arange(
            1,
            2 * N,
            2,
        )
    )
    A = 0.5 * (
        P[:, None] * P[None, :]
    )  # Outer product (equivalent to broadcasting)
    A = torch.tril(A, diagonal=-1)
    A = A + torch.diag(torch.arange(1, N + 1))

    return -A


def compute_S4DInv(N):
    """
    Constructs the S4D-Inv matrix A, represented as a 1-D torch.Tensor.

    :param int hidden_dim: The size of the matrix.
    :return: A matrix initialized using the S4D-Inv method.
    :rtype: torch.Tensor
    """
    n = torch.arange(N, dtype=torch.float32)
    return -0.5 + 1j * (N / torch.pi) * (N / (2 * n + 1) - 1)


def compute_S4DLin(N):
    """
    Constructs the S4D-Lin hidden-to-hidden matrix A.
    :param int N: The size of the matrix.
    :return: A matrix initialized using the S4D-Inv method.
    :rtype: torch.Tensor
    """
    n = torch.arange(N, dtype=torch.float32)
    return -0.5 + 1j * (n * torch.pi)


def compute_S4DQuad(N):
    """
    Constructs the S4D-Quad hidden-to-hidden matrix A.
    :param int N: The size of the matrix.
    :return: A matrix initialized using the S4D-Inv method.
    :rtype: torch.Tensor
    """
    n = torch.arange(N, dtype=torch.float32)
    return 1 / torch.pi * (1 + 2 * n) ** 2


def compute_S4DReal(N):
    """
    Constructs the S4D-Real hidden-to-hidden matrix A.
    :param int N: The size of the matrix.
    :return: A matrix initialized using the S4D-Inv method.
    :rtype: torch.Tensor
    """
    return -(torch.rand(N) + 1)


def compute_dplr(A):
    """
    TODO
    """
    # Compute p and q in a vectorized manner
    N = A.shape[0]
    indices = torch.arange(1, N + 1, dtype=torch.float32)
    p = 0.5 * torch.sqrt(2 * indices + 1.0)
    q = 2 * p
    # Construct S efficiently
    S = A + p[:, None] * q[None, :]
    Lambda, V = torch.linalg.eig(S)
    Vc = V.conj().T
    p, q = p.to(Vc.dtype), q.to(Vc.dtype)
    p = Vc @ p
    q = Vc @ q
    return Lambda, p, q
