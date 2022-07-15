import torch
from tabulate import tabulate
from torch import nn, Tensor

from modules.layers.conv_group import ConvGroup


class UGroup(nn.Module):
    """
    [channels: in_c, s] -> UGroup -> [channels: out_c, s]
    """

    def __init__(self, in_c: int, out_c: int, dim: int, k_size: int, use_bn: bool, use_rs: bool):
        super().__init__()

        # conv_1: [in_c, s] -> [d, s]
        self.conv_1 = ConvGroup(nn.Conv2d(in_c, dim, kernel_size=k_size, padding='same'), use_bn=use_bn)

        # down sample (pool): [d, s] -> [d, s/2]
        self.ds = nn.MaxPool2d(2, stride=2, ceil_mode=True) if use_rs else nn.Identity()

        # conv_2: [d, s/2] -> [d, s/2]
        self.conv_2 = ConvGroup(nn.Conv2d(dim, dim, kernel_size=k_size, padding='same'), use_bn=use_bn)

        # dilated conv: [d, s/2] -> [d, s/2]
        self.conv_dil = ConvGroup(nn.Conv2d(dim, dim, kernel_size=k_size, padding='same', dilation=2), use_bn=use_bn)

        # conv_3: [2d, s/2] -> [d, s/2]
        self.conv_3 = ConvGroup(nn.Conv2d(2 * dim, dim, kernel_size=k_size, padding='same'), use_bn=use_bn)

        # up sample: [d, s/2] -> [d, s]
        self.us = nn.Upsample(scale_factor=2, mode='bilinear') if use_rs else nn.Identity()

        # conv_4: [2d, s] -> [out_c, s]
        self.conv_4 = ConvGroup(nn.Conv2d(2 * dim, out_c, kernel_size=k_size, padding='same'), use_bn=use_bn)

    def forward(self, x: Tensor) -> Tensor:
        f_in = x
        # conv_1: [in_c, s] -> [d, s]
        f_1 = self.conv_1(f_in)
        # conv_2: [d, s/2] -> [d, s/2]
        f_t = self.ds(f_1)
        f_2 = self.conv_2(f_t)
        # conv_dil: [d, s/2] -> [d, s/2]
        f_d = self.conv_dil(f_2)
        # conv_3: [2d, s/2] -> [d, s/2]
        f_t = torch.cat((f_2, f_d), dim=1)
        f_3 = self.conv_3(f_t)
        # conv_4: [2d, s] -> [out_c, s]
        f_t = self.us(f_3)
        f_t = torch.cat([f_1, f_t], dim=1)
        f_out = self.conv_4(f_t)
        return f_out

    def __str__(self):
        table = [[n, p.mean(), p.grad.mean()] for n, p in self.named_parameters() if p.grad is not None]
        return tabulate(table, headers=['layer', 'weights', 'grad'], tablefmt='pretty')
