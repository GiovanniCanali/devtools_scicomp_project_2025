import torch
from torch.nn import LayerNorm
from .block import H3Block


class H3(torch.nn.Module):
    r"""
    Implementation of the H3 model.

    This model is a stack of H3 blocks, which leverage the S4 shift block and
    the S4 diagonal block to create a hybrid architecture that mimics the
    behavior of the attention mechanism while maintaining the efficiency of
    state space models. Between the blocks, a linear layer and an activation
    function are applied, together with an optional normalization layer.

    The H3 model is designed for efficiently modeling long-range dependencies
    in sequential data using structured state space representations.
    It enables improved scalability and performance compared to traditional
    recurrent architectures.

    .. seealso::
        **Original Reference**: Fu, D., Dao, T., et al. (2023).
        "Hungry Hungry Hippos: Towards Language Modeling with State Space
        Models".
        arXiv:2212.14052.
        DOI: `<https://doi.org/10.48550/arXiv.2212.14052>_`.
    """

    def __init__(
        self,
        model_dim,
        hid_dim,
        method,
        heads,
        dt=0.1,
        initialization="S4D-Inv",
        discretization="bilinear",
        n_layers=2,
        normalization=False,
        activation=torch.nn.ReLU,
        real_random=False,
        imag_random=False,
        **kwargs,
    ):
        """
        Initialization of the H3 model.

        :param int model_dim: The input dimension.
        :param int hid_dim: The hidden state dimension.
        :param str method: The forward computation method. Available options
            are: recurrent, convolutional.
        :param int heads: The number of attention heads. It must be a divisor of
            the input dimension.
        :param float dt: The time step for discretization. Default is `0.1`.
        :param str initialization: The method for initializing the A matrix in
            the diagonal S4 block. Options are: S4D-Inv, S4D-Lin, S4D-Quad,
            S4D-Real, real, complex. Default is `"S4D-Inv"`.
        :param str discretization: The method for discretizing the dynamics of
            the diagonal S4 block.
            Options are: bilinear, zoh. Default is `"bilinear"`.
        :param int n_layers: Number of H3 blocks. Default is 2.
        :param bool normalization: Whether to apply layer normalization.
            Default is `False`.
        :param torch.nn.Module activation: The activation function.
            Default is `torch.nn.ReLU`.
        :param bool real_random: If `True`, the real part of the A matrix of the
            diagonal block is initialized at random between 0 and 1.
            Default is `False`.
        :param bool imag_random: If `True`, the imaginary part of the A matrix
            of the diagonal block is initialized at random between 0 and 1.
            Default is `False`.
        :param dict kwargs: Additional arguments for the class constructor.
        """
        super().__init__()

        # Initialize the layers
        layers = []
        for _ in range(n_layers):
            if normalization:
                layers.append(LayerNorm(model_dim, elementwise_affine=False))
            layers.append(
                H3Block(
                    model_dim=model_dim,
                    hid_dim=hid_dim,
                    method=method,
                    heads=heads,
                    dt=dt,
                    initialization=initialization,
                    discretization=discretization,
                    real_random=real_random,
                    imag_random=imag_random,
                    **kwargs,
                )
            )
            layers.append(activation())
            layers.append(torch.nn.Linear(model_dim, model_dim))

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the H3 model.

        :param torch.Tensor x: The input tensor.
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        return self.layers(x)
