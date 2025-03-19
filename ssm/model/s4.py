import torch
from .block.s4_base_block import S4Block


class S4(torch.nn.Module):
    def __init__(
        self, method, input_dim, output_dim, hidden_dim, hippo=True, fixed=False
    ):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [
                S4Block(method=method, hidden_dim=hidden_dim, hippo=hippo)
                for _ in range(input_dim)
            ]
        )
        self.mixing_fc = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x has shape (B, L, H)
        y = []
        for i, block in enumerate(self.blocks):
            y.append(block(x[..., i]))
        y = torch.cat(y, dim=-1)

        y = self.mixing_fc(y)
        return y

    def change_forward(self, method):
        for block in self.blocks:
            block.change_forward(method)
