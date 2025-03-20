import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils import compute_hippo


class S4AdvancedBlock(nn.Module):
    def __new__(cls, method, **kwargs):
        if method == "recurrent":
            return S4RecurrentBlock(**kwargs)
        elif method == "fourier":
            raise NotImplementedError
        else:
            raise ValueError(f"Method {method} not recognized")


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
