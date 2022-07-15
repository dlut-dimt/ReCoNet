import torch
from torch import nn, Tensor

from modules.layers.conv_group import ConvGroup


class UNet(nn.Module):
    """
    An u-net architecture.
    [channels: in_c, s] -> UNet -> [channels: out_c, s]

    default:
    encoder: [16, 32, 32, 32]
    decoder: [32, 32, 32, 32, 32, 16]
    """

    def __init__(self, in_c: int, out_c: int):
        super().__init__()

        # encoder and decoder feature
        self.enc_c = [16, 32, 32, 32]
        self.dec_c = [32, 32, 32, 32, 32, 16, out_c]

        # upsample
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        self.encoder = nn.ModuleList()
        prev_c = in_c
        for c in self.enc_c:
            # [size: s] -> [size: s/2]
            self.encoder.append(ConvGroup(nn.Conv2d(prev_c, c, kernel_size=3, stride=2, padding=1), use_bn=False))
            prev_c = c

        # configure decoder (up-sampling path)
        rev_enc_c = list(reversed(self.enc_c))
        self.decoder = nn.ModuleList()
        for i, c in enumerate(self.dec_c[:len(self.enc_c)]):
            tmp_c = prev_c + rev_enc_c[i] if i > 0 else prev_c
            self.decoder.append(ConvGroup(nn.Conv2d(tmp_c, c, kernel_size=3, padding='same'), use_bn=False))
            prev_c = c

        # configure decoder suffix (no up-sampling)
        prev_c += in_c
        self.suffix = nn.ModuleList()
        for c in self.dec_c[len(self.enc_c):]:
            self.suffix.append(ConvGroup(nn.Conv2d(prev_c, c, kernel_size=3, padding='same'), use_bn=False))
            prev_c = c

    def forward(self, x: Tensor) -> Tensor:
        # encoder
        f_in = [x]
        for layer in self.encoder:
            f_in.append(layer(f_in[-1]))

        # decoder: conv -> upsample -> concat
        f_x = f_in.pop()
        for layer in self.decoder:
            f_x = layer(f_x)
            f_x = self.upsample(f_x)
            f_x = torch.cat([f_x, f_in.pop()], dim=1)

        # suffix
        for layer in self.suffix:
            f_x = layer(f_x)

        return f_x
