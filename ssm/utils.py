import torch


def compute_hippo(N):
    """
    Constructs the HIPPO hidden-to-hidden matrix A.

    :param int N: The size of the HIPPO matrix.
    :return: A (N, N) matrix initialized using the HIPPO method.
    :rtype: torch.Tensor
    """
    A = torch.zeros(N, N)

    # Compute square roots
    idx = torch.arange(N)
    sqrt_terms = (2 * idx + 1).sqrt()

    # Compute outer product
    A = sqrt_terms[:, None] * sqrt_terms[None, :]

    # Zero out the upper triangular part
    A = torch.tril(A)

    # Set diagonal to n + 1
    A.diagonal().copy_(idx + 1)

    return -A


def compute_S4DInv(N):
    """
    Construct the S4D-Inv matrix A.

    :param int N: The size of the matrix.
    :return: The computed matrix A.
    :rtype: torch.Tensor
    """
    n = torch.arange(N, dtype=torch.float32)
    return -0.5 + 1j * (N / torch.pi) * (N / (2 * n + 1) - 1)


def compute_S4DLin(N):
    """
    Construct the S4D-Lin matrix A.

    :param int N: The size of the matrix.
    :return: The computed matrix A.
    :rtype: torch.Tensor
    """
    n = torch.arange(N, dtype=torch.float32)
    return -0.5 + 1j * (n * torch.pi)


def compute_S4DQuad(N):
    """
    Construct the S4D-Quad matrix A.

    :param int N: The size of the matrix.
    :return: The computed matrix A.
    :rtype: torch.Tensor
    """
    n = torch.arange(N, dtype=torch.float32)
    return 1 / torch.pi * (1 + 2 * n) ** 2


def compute_S4DReal(N):
    """
    Construct the S4D-Real matrix A.

    :param int N: The size of the matrix.
    :return: The computed matrix A.
    :rtype: torch.Tensor
    """
    return -(torch.rand(N) + 1)


def compute_dplr(A):
    """
    Construct the diagonal plus low-rank (DPLR) form of matrix A. The matrix A
    is decomposed into a diagonal matrix Lambda and in a low-rank matrix given
    by the outer product of two vectors p and q.

    :param torch.Tensor A: The input matrix.
    :return: The diagonal plus low-rank form of A.
    :rtype: tuple
    """
    # Initialize p and q
    idx = torch.arange(1, A.shape[0] + 1, dtype=torch.float32)
    p = 0.5 * torch.sqrt(2 * idx + 1.0)
    q = 2 * p

    # Construct a matrix S
    S = A + p[:, None] * q[None, :]

    # Compute Lambda, p, q
    Lambda, V = torch.linalg.eig(S)
    Vc = V.conj().T
    p = Vc @ p.to(Vc.dtype)
    q = Vc @ q.to(Vc.dtype)
    return Lambda, p, q
