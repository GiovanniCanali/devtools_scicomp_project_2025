import warnings
import torch
from . import S4BaseBlock, S4DBlock, S4LowRankBlock, S6Block


class MambaBlock(torch.nn.Module):
    """
    Implementation of the Mamba block. It combines a linear layer,
    a convolutional layer, and an SSM block as explained in the original
    Mamba paper and in the official repository.

    .. seealso::
        **Original Reference**: Gu, A. and Dao, T. (2024).
        "Mamba: Linear-Time Sequence Modeling with Selective State Spaces".
        arXiv:2312.00752.
        DOI: `<https://doi.org/10.48550/arXiv.2312.00752>_`.
        **Official GitHub Repository**:
        `<https://github.com/state-spaces/mamba>`_.
    """

    def __init__(
        self,
        input_dim,
        expansion_factor,
        kernel_size,
        normalization=False,
        ssm_type="S4",
        **kwargs,
    ):
        """
        Initializes the Mamba block with the specified parameters.

        :param int input_dim: The input dimension.
        :param int expansion_factor: The expansion factor for the input
            dimension.
        :param int kernel_size: The kernel size for the convolutional layer.
        :param bool normalization: Whether to apply layer normalization.
            Default is `False`.
        :param str ssm_type: The type of SSM block to use. Available options
            are: `"S4"`, `"S4D"`, `"S4LowRank"`, `"S6"`. Default is `"S4"`.
        :param dict kwargs: Additional arguments for the SSM block constructor.
        :raises ValueError: If an invalid `ssm_type` is provided.
        :raises RuntimeError: If an invalid `ssm_type` is provided.
        """
        super().__init__()

        if "input_dim" in kwargs:
            warnings.warn(
                "input_dim is determined by the input_net, ignoring "
                "kwargs['input_dim']"
            )
            kwargs.pop("input_dim")
        kwargs["input_dim"] = expansion_factor * input_dim
        if "hid_dim" in kwargs:
            warnings.warn(
                "hidden_dim is determined by the input_net, ignoring "
                "kwargs['hidden_dim']"
            )
            kwargs.pop("hidden_dim")
        kwargs["hid_dim"] = expansion_factor * input_dim

        self.input_net = torch.nn.Linear(
            input_dim, expansion_factor * input_dim
        )
        self.input_net_res = torch.nn.Linear(
            input_dim, expansion_factor * input_dim
        )
        self.output_net = torch.nn.Linear(
            expansion_factor * input_dim, input_dim
        )
        self.ssm = self._initialize_ssm_block(ssm_type, **kwargs)
        self.silu = torch.nn.SiLU()
        self.conv1d = torch.nn.Conv1d(
            in_channels=expansion_factor * input_dim,
            out_channels=expansion_factor * input_dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
        )
        if normalization:
            self.norm = torch.nn.LayerNorm(
                expansion_factor * input_dim, elementwise_affine=False
            )
        else:
            self.norm = None

    def forward(self, x):
        """
        Forward pass of the Mamba block.
        :param torch.Tensor x: The input tensor with shape `(B, L, H)`.
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        x, x_res = self.input_net(x), self.silu(self.input_net_res(x))
        x = self.conv1d(x.transpose(1, 2))[:, :, : x.shape[1]].transpose(1, 2)
        x = self.silu(x)
        x = self.ssm(x)
        x = x + x_res

        if self.norm is not None:
            x = self.norm(x)

        x = self.output_net(x)
        return x

    def _initialize_ssm_block(self, ssm_type, **kwargs):
        """
        Initialize the SSM block based on the specified type.
        :param str ssm_type: The type of SSM block to use. Available options
            are: `"S4"`, `"S4D"`, `"S4LowRank"`, `"S6"`.
        :param dict kwargs: Additional arguments for the SSM block constructor.
        :raises ValueError: If an invalid `ssm_type` is provided.
        """
        if ssm_type == "S4":
            return S4BaseBlock(**kwargs)
        elif ssm_type == "S4D":
            return S4DBlock(**kwargs)
        elif ssm_type == "S4LowRank":
            return S4LowRankBlock(**kwargs)
        elif ssm_type == "S6":
            return S6Block(**kwargs)
        else:
            raise ValueError(f"Unknown SSM type: {ssm_type}")
