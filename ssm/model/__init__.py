"""
Module for the definition of the models.
"""

__all__ = ["S4", "S6", "Mamba", "GatedMLP", "Transformer", "LSTM"]


from .s4 import S4
from .s6 import S6
from .lstm import LSTM
from .mamba import Mamba
from .gated_mlp import GatedMLP
from .transformer import Transformer
