from kornia import create_meshgrid
from torch import Tensor
from torch.nn import functional


def transformer(i_in: Tensor, flow: Tensor) -> [Tensor, Tensor]:
    # create mesh grid: [1, h, w, 2]
    h, w = flow.size()[1:3]
    grid = create_meshgrid(height=h, width=w, normalized_coordinates=False, device=flow.device).to(flow.dtype)
    # new locations: [b, h, w, 2]
    locs = grid + flow
    # normalize
    locs[..., 0] = (locs[..., 0] / (w - 1) - 0.5) * 2
    locs[..., 1] = (locs[..., 1] / (h - 1) - 0.5) * 2
    # apply transform
    i_out = functional.grid_sample(i_in, locs, align_corners=True, mode='bilinear')
    # return moved image and flow
    return i_out, locs
