import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class S4AdvancedBlock(nn.Module):
    def __new__(cls, method, **kwargs):
        if method == "recurrent":
            return S4RecurrentBlock(**kwargs)
        elif method == "fourier":
            raise NotImplementedError
        else:
            raise ValueError(f"Method {method} not recognized")


def hippo(N):
    """
    Defining HIPPO matrix for the S4 block.
    :param int N: Shape of the matrix.
    :return: A matrix of shape (N, N) initialized according to the rules
        defined in the paper.
    :rtype: torch.Tensor
    """
    P = torch.sqrt(torch.arange(1, 2 * N, 2, dtype=torch.float32))
    A = 0.5 * (P[:, None] * P[None, :])
    A = torch.tril(A, diagonal=-1) - torch.diag(
        torch.arange(N, dtype=torch.float32)
    )
    return -A


class S4RecurrentBlock(nn.Module):
    """
    S4 block with recurrent/sequential computation.

    :param int hidden_dim: Dimension of the hidden state.
    :param float dt: Time step.
    :param bool hippo: Whether to use the HIPPO matrix.
    :param bool fixed: Whether to fix the parameters.
    """

    def __init__(
        self,
        hidden_dim: int,
        dt: float = 0.1,
        hippo: bool = False,
        fixed: bool = False,
    ):
        super().__init__()
        if hippo:
            A = hippo(hidden_dim)
        else:
            A = torch.rand(hidden_dim, hidden_dim)
