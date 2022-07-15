import torch
from tabulate import tabulate
from torch import nn, Tensor

from modules.layers.conv_group import ConvGroup


class DGroup(nn.Module):
    """
    [channels: dim, s] -> DGroup -> [channels: 1, s]
    """

    def __init__(self, in_c: int, out_c: int, dim: int, k_size: int, use_bn: bool):
        super().__init__()

        # conv_d: [dim] -> [1]
        self.conv_d = nn.ModuleList([
            ConvGroup(nn.Conv2d(in_c, dim, kernel_size=k_size, padding='same', dilation=(i + 1)), use_bn=use_bn)
            for i in range(3)
        ])

        # conv_s: [3] -> [1]
        self.conv_s = nn.Sequential(
            nn.Conv2d(3 * dim, out_c, kernel_size=3, padding='same'),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        f_in = x
        # conv_d
        f_x = [conv(f_in) for conv in self.conv_d]
        # suffix
        f_t = torch.cat(f_x, dim=1)
        f_out = self.conv_s(f_t)
        return f_out

    def __str__(self):
        table = [[n, p.mean(), p.grad.mean()] for n, p in self.named_parameters() if p.grad is not None]
        return tabulate(table, headers=['layer', 'weights', 'grad'], tablefmt='pretty')
