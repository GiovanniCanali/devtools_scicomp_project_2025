import math
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


def compute_S4DInv(N, real_random=False, imag_random=False):
    """
    Construct the S4D-Inv matrix A.

    :param int N: The size of the matrix.
    :param bool real_random: If `True`, the real part of the A matrix is
        initialized at random between 0 and 1. Default is `False`.
    :param bool imag_random: If `True`, the imaginary part of the A matrix
        is initialized at random between 0 and 1. Default is `False`.
    :return: The computed matrix A.
    :rtype: torch.Tensor
    """
    if real_random:
        real_part = -torch.rand(N, dtype=torch.float32)
    else:
        real_part = -torch.ones(N, dtype=torch.float32) * 1 / 2
    if imag_random:
        imag_part = torch.rand(N, dtype=torch.float32)
    else:
        imag_part = torch.arange(N, dtype=torch.float32)
    return real_part + 1j * (N / torch.pi) * (N / (2 * imag_part + 1) - 1)


def compute_S4DLin(N, real_random=False, imag_random=False):
    """
    Construct the S4D-Lin matrix A.

    :param int N: The size of the matrix.
    :param bool real_random: If `True`, the real part of the A matrix is
        initialized at random between 0 and 1. Default is `False`.
    :param bool imag_random: If `True`, the imaginary part of the A matrix
        is initialized at random between 0 and 1. Default is `False`.
    :return: The computed matrix A.
    :rtype: torch.Tensor
    """
    if real_random:
        real_part = -torch.rand(N, dtype=torch.float32)
    else:
        real_part = -torch.ones(N, dtype=torch.float32) * 1 / 2
    if imag_random:
        imag_part = torch.rand(N, dtype=torch.float32)
    else:
        imag_part = torch.arange(N, dtype=torch.float32)
    return real_part + 1j * (imag_part * torch.pi)


def compute_S4DQuad(N, real_random=False):
    """
    Construct the S4D-Quad matrix A.

    :param int N: The size of the matrix.
    :param bool real_random: If `True`, the real part of the A matrix is
        initialized at random between 0 and 1. Default is `False`.
    :return: The computed matrix A.
    :rtype: torch.Tensor
    """
    if real_random:
        real_part = torch.rand(N, dtype=torch.float32)
    else:
        real_part = torch.arange(N, dtype=torch.float32)
    return 1 / torch.pi * (1 + 2 * real_part) ** 2


def compute_S4DReal(N, real_random=False):
    """
    Construct the S4D-Real matrix A.

    :param int N: The size of the matrix.
    :param bool real_random: If `True`, the real part of the A matrix is
        initialized at random between 0 and 1. Default is `False`.
    :return: The computed matrix A.
    :rtype: torch.Tensor
    """
    if real_random:
        real_part = torch.rand(N, dtype=torch.float32)
    else:
        real_part = torch.arange(N, dtype=torch.float32)
    return -(real_part + 1)


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


def initialize_dt(input_dim, dt_min, dt_max, inverse_softplus=False):
    """
    Initialize the time step dt for the S4 and S6 blocks.

    :param float dt_min: The minimum time step for discretization.
    :param float dt_max: The maximum time step for discretization.
    :return: Initialized time step dt tensor of shape (input_dim,).
    :rtype: torch.Tensor
    """

    # Sample dt from a uniform distribution on [dt_min, dt_max]
    dt = torch.exp(
        torch.rand(input_dim) * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    ).clamp(min=1e-4)

    # Apply the inverse softplus to dt
    dt = dt + torch.log(-torch.expm1(-dt)) if inverse_softplus else dt

    return dt
