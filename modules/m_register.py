import torch
from torch import nn, Tensor

from modules.layers.conv_group import ConvGroup
from modules.layers.d_group import DGroup
from modules.layers.u_group import UGroup


class MRegister(nn.Module):
    def __init__(self, in_c: int, dim: int, k_sizes: tuple, use_bn: bool, use_rs: bool):
        super().__init__()

        # stem: [2] -> [d]
        self.stem = ConvGroup(nn.Conv2d(in_c, dim, kernel_size=3, padding='same'), use_bn=use_bn)

        # group s: [d, 2d, 3d] -> [d]
        self.groups = nn.ModuleList(
            [
                UGroup(in_c=(i + 1) * dim, out_c=dim, dim=dim, k_size=k_sizes[i], use_bn=use_bn, use_rs=use_rs)
                for i in range(3)
            ]
        )

        # decoder: [3d] -> [2]
        self.decoder = DGroup(in_c=3 * dim, out_c=2, dim=dim, k_size=3, use_bn=use_bn)
        self.decoder.conv_s = nn.Conv2d(3 * dim, 2, kernel_size=3, padding='same')

        # init flow layer with small weights and bias
        self.decoder.conv_s.apply(self.init_weights)

    def forward(self, i_in: Tensor):
        # stem: [2] -> [d]
        f_x = self.stem(i_in)

        # group: [d, 2d, 3d] -> [3d]
        f_0 = self.groups[0](f_x)
        f_1 = self.groups[1](torch.cat([f_0, f_x], dim=1))
        f_2 = self.groups[2](torch.cat([f_0, f_1, f_x], dim=1))
        f_i = torch.cat([f_0, f_1, f_2], dim=1)  # [b, 3d, h, w]

        # decoder: [3d] -> [2]
        flow = self.decoder(f_i).permute(0, 2, 3, 1)  # [b, h, w, 2]

        # return middle vars during forward process
        return flow

    @staticmethod
    @torch.no_grad()
    def init_weights(m):
        if type(m) == nn.Conv2d:
            nn.init.normal_(m.weight, mean=0, std=1e-5)
            nn.init.zeros_(m.bias)
