from torch import nn, Tensor


class ConvGroup(nn.Module):
    def __init__(self, conv: nn.Conv2d, use_bn: bool):
        super().__init__()

        # (Conv2d, BN, GELU)
        dim = conv.out_channels
        self.group = nn.Sequential(
            conv,
            nn.BatchNorm2d(dim) if use_bn else nn.Identity(),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.group(x)
