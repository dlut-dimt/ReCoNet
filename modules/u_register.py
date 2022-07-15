import torch
from torch import nn, Tensor

from modules.layers.u_net import UNet


class URegister(nn.Module):
    def __init__(self, in_c: int, dim: int):
        super().__init__()

        # u-net core: [2] -> [16]
        self.unet = UNet(in_c=in_c, out_c=dim)

        # flow layer
        self.flow = nn.Conv2d(dim, 2, kernel_size=3, padding='same')

        # init flow layer
        self.flow.apply(self.init_weights)

    def forward(self, i_in: Tensor):
        # unet: [2] -> [d]
        f_x = self.unet(i_in)

        # flow: [b, d, h, w] -> [b, h, w, 2]
        flow = self.flow(f_x).permute(0, 2, 3, 1)

        # return middle vars during forward process
        return flow

    @staticmethod
    @torch.no_grad()
    def init_weights(m):
        if type(m) == nn.Conv2d:
            nn.init.normal_(m.weight, mean=0, std=1e-5)
            nn.init.zeros_(m.bias)
