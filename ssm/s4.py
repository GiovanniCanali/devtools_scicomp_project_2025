import torch

class S4_layer(torch.nn.Module):

    def __init__(self, latent_dim: int, input_dim: int, output_dim: int = None):
        # Dimensions
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim

        # Parameters
        self.A = torch.nn.Parameter(torch.rand(self.latent_dim, self.latent_dim))
        self.B = torch.nn.Parameter(torch.rand(self.latent_dim, ))
        self.C = torch.nn.Parameter(torch.rand(1, self.latent_dim))

        # Latent state
        #self.h = torch.zeros(self.latent_dim)

    def forward(self, x):
        # x has shape [B, L, D]
        sequence_length = x.shape[0]
        x = x.permute(1, 0, 2)  # [L, B, D]
        # create empty tensor for y and h
        y = torch.empty(x.shape[0], x.shape[1], self.output_dim)
        h = torch.empty(x.shape[0]+1, x.shape[1], self.latent_dim)

        # Notice that h times are shifted by one (i.e h[0] is time -1)
        h[0] = torch.zeros(x.shape[1], self.latent_dim)
        for t in range(sequence_length):
            h[t+1] = self.A @ h[t] + self.B @ x[t]
            y[t] = self.C @ h[t+1]

        return y, h
