import torch


class Transformer(torch.nn.Module):
    """
    Wrapper class for torch.nn.TransformerEncoder.
    """

    def __init__(
        self,
        model_dim,
        hidden_dim,
        heads,
        n_layers=2,
        dropout=0.1,
        activation="gelu",
        **kwargs,
    ):
        """
        Initialization of the Transformer model.

        :param int model_dim: The input dimension.
        :param int hidden_dim: The hidden state dimension.
        :param int heads: The number of attention heads.
        :param int n_layers: Number of encoder layers. Default is 2.
        :param float dropout: Dropout rate. Default is 0.1.
        :param str activation: The activation function. Must be one of
            `'gelu'` or `'relu'`. Default is `'gelu'`.
        :param dict kwargs: Additional keyword arguments used in the model.
        :raises ValueError: If the number of heads is not a divisor of the
            input dimension.
        """
        super().__init__()

        # Check if the number of heads is a divisor of the input dimension
        if model_dim % heads != 0:
            raise ValueError(
                "The number of heads must be a divisor of the input dimension."
            )

        # Initialize the transformer encoder layer
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )

        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=n_layers,
            norm=torch.nn.LayerNorm(model_dim),
        )

    def forward(self, x):
        """
        Forward pass through the Transformer model with optional masks.

        :param torch.Tensor x: Input tensor (batch, seq length, dimension).
        :return: Output tensor.
        :rtype: torch.Tensor
        """
        return self.transformer_encoder(x)
