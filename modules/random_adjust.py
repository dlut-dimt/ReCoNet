import torch
from kornia import create_meshgrid
from kornia.augmentation import RandomPerspective, RandomElasticTransform
from kornia.filters import get_gaussian_kernel2d, filter2d
from kornia.geometry import normalize_homography, transform_points, get_perspective_transform
from kornia.utils.helpers import _torch_inverse_cast
from torch import nn, Tensor


class RandomAdjust(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # elastic
        ks, sigma = config['kernel_size'], config['sigma']
        re = RandomElasticTransform(kernel_size=ks, sigma=sigma, p=1)
        self.re = re

        # perspective
        ds = config['distortion_scale']
        rp = RandomPerspective(distortion_scale=ds, p=1)
        self.rp = rp

        # swap params
        self.size = ()
        self.device, self.dtype = torch.device('cpu'), torch.float

    def forward(self, x: Tensor) -> (Tensor, dict):
        # params
        self.size = x.size()
        self.device, self.dtype = x.device, x.dtype
        B, _, H, W = x.size()

        params = {}

        # elastic
        if 'e' in self.config['transforms']:
            x = self.re(x)
            noise = self.re._params['noise'].to(self.device)  # [b, h, w, 2]
            disp_e = self.get_elastic_disp(noise)  # [b, h, w, 2]
            # rebase
            # disp_e = disp_e.permute(0, 3, 1, 2)  # [b, 2, h, w]
            # disp_e = self.re.apply_transform(disp_e, self.re._params)
            # disp_e = disp_e.permute(0, 2, 3, 1)  # [b, h, w, 2]
            params |= {'de': disp_e}

        # perspective
        if 'p' in self.config['transforms']:
            # generate params
            self.rp(x)
            # fix end_points
            corner = self.rp._params['start_points']
            self.rp._params['start_points'] = self.rp._params['end_points']
            self.rp._params['end_points'] = corner
            # transform
            x = self.rp(x, params=self.rp._params)
            # calculate offset disp
            f, t = self.rp._params['start_points'].to(x), self.rp._params['end_points'].to(x)
            matrix = get_perspective_transform(t, f)  # matrix end_points -> start_points
            disp_p = self.get_perspective_disp(matrix)  # [b, h, w, 2]
            params |= {'dp': -disp_p}

        return x, params

    def get_perspective_disp(self, transform: Tensor) -> Tensor:
        # params
        B, _, H, W = self.size
        h_out, w_out = H, W

        # we normalize the 3x3 transformation matrix and convert to 3x4
        dst_norm_trans_src_norm = normalize_homography(transform, (H, W), (h_out, w_out))  # Bx3x3

        src_norm_trans_dst_norm = _torch_inverse_cast(dst_norm_trans_src_norm)  # Bx3x3

        # this piece of code substitutes F.affine_grid since it does not support 3x3
        grid = create_meshgrid(h_out, w_out, normalized_coordinates=True, device=self.device).to(self.dtype)
        grid = grid.repeat(B, 1, 1, 1)
        disp = transform_points(src_norm_trans_dst_norm[:, None, None], grid) - grid  # disp: infrared -> \bar{infrared}
        return disp

    def get_elastic_disp(self, noise: Tensor) -> Tensor:
        # params
        config = self.config
        ks, sigma = config['kernel_size'], config['sigma']

        # Get Gaussian kernel for 'visible' and 'infrared' displacement
        kernel_x = get_gaussian_kernel2d(ks, sigma)[None]
        kernel_y = get_gaussian_kernel2d(ks, sigma)[None]

        # Convolve over a random displacement matrix and scale them with 'alpha'
        disp_x = noise[:, :1]
        disp_y = noise[:, 1:]

        disp_x = filter2d(disp_x, kernel=kernel_y, border_type="constant")
        disp_y = filter2d(disp_y, kernel=kernel_x, border_type="constant")

        # stack and normalize displacement
        disp = torch.cat([disp_x, disp_y], dim=1).permute(0, 2, 3, 1)  # disp: infrared -> \bar{infrared}
        return disp
