import torch
from torch import nn, Tensor

from modules.layers.d_group import DGroup


class Fuser(nn.Module):
    def __init__(self, depth: int, dim: int, use_bn: bool):
        super().__init__()
        self.depth = depth

        # attention layer: [2] -> [1], [2] -> [1]
        self.att_a_conv = nn.Conv2d(2, 1, kernel_size=3, padding='same', bias=False)
        self.att_b_conv = nn.Conv2d(2, 1, kernel_size=3, padding='same', bias=False)

        # dilation fuse
        self.decoder = DGroup(in_c=3, out_c=1, dim=dim, k_size=3, use_bn=use_bn)

    def forward(self, i_in: Tensor, init_f: str = 'max', show_detail: bool = False):
        # recurrent subnetwork
        # generate f_0 with initial function
        i_1, i_2 = torch.chunk(i_in, chunks=2, dim=1)
        i_f = [torch.max(i_1, i_2) if init_f == 'max' else (i_1 + i_2) / 2]
        att_a, att_b = [], []

        # loop in subnetwork
        for _ in range(self.depth):
            i_f_x, att_a_x, att_b_x = self._sub_forward(i_1, i_2, i_f[-1])
            i_f.append(i_f_x), att_a.append(att_a_x), att_b.append(att_b_x)

        # return as expected
        return (i_f, att_a, att_b) if show_detail else i_f[-1]

    def _sub_forward(self, i_1: Tensor, i_2: Tensor, i_f: Tensor):
        # attention
        att_a = self._attention(self.att_a_conv, i_1, i_f)
        att_b = self._attention(self.att_b_conv, i_2, i_f)

        # focus on attention
        i_1_w = i_1 * att_a
        i_2_w = i_2 * att_b

        # dilation fuse
        i_in = torch.cat([i_1_w, i_f, i_2_w], dim=1)
        i_out = self.decoder(i_in)

        # return fusion result of current recurrence
        return i_out, att_a, att_b

    @staticmethod
    def _attention(att_conv, i_a, i_b):
        i_in = torch.cat([i_a, i_b], dim=1)
        i_max, _ = torch.max(i_in, dim=1, keepdim=True)
        i_avg = torch.mean(i_in, dim=1, keepdim=True)
        i_in = torch.cat([i_max, i_avg], dim=1)
        i_out = att_conv(i_in)
        return torch.sigmoid(i_out)
