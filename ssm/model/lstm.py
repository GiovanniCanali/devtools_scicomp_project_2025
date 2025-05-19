import torch


class LSTM(torch.nn.Module):
    """
    Wrapper class for torch.nn.LSTM.
    """

    def __init__(
        self,
        model_dim,
        hidden_dim,
        n_layers=2,
        dropout=0.1,
        **kwargs,
    ):
        """
        Initialization of the LSTM model.

        :param int model_dim: The input dimension.
        :param int hidden_dim: The hidden state dimension.
        :param int n_layers: Number of LSTM layers. Default is 2.
        :param float dropout: Dropout rate. Default is 0.1.
        :param dict kwargs: Additional keyword arguments used in the model.
        """
        super().__init__()

        # Initialize the lstm model
        self.lstm = torch.nn.LSTM(
            input_size=model_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
        )

        # Linear layer to project the output to the input dimension
        self.linear = torch.nn.Linear(hidden_dim, model_dim)

    def forward(self, x):
        """
        Forward pass through the LSTM model.

        :param torch.Tensor x: Input tensor (batch, seq length, dimension).
        :return: Output tensor.
        :rtype: torch.Tensor
        """
        output, _ = self.lstm(x)
        return self.linear(output)
