from argparse import Namespace

import pytorch_lightning as pl
import torch
from kornia.filters import canny
from kornia.losses import ssim_loss
from torch import Tensor
from torch.nn.functional import mse_loss, l1_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau

from modules.functions.integrate import integrate
from modules.functions.transformer import transformer
from modules.fuser import Fuser
from modules.m_register import MRegister
from modules.u_register import URegister


class ReCo(pl.LightningModule):
    def __init__(self, args: Namespace):
        super().__init__()
        self.save_hyperparameters()
        self.dim = args.dim

        # init register
        match args.register:
            case 'm':
                print('Init with m-register')
                self.register = MRegister(in_c=4, dim=16, k_sizes=(3, 5, 7), use_bn=False, use_rs=True)
            case 'u':
                print('Init with u-register')
                self.register = URegister(in_c=4, dim=16)
            case _:
                print('Turn off register')
                self.register = None

        # init fuser
        self.fuser = Fuser(depth=3, dim=self.dim, use_bn=False)

        # learning rate
        self.lr = args.lr

        # weight
        self.rf_weight = args.rf_weight
        self.r_weight, self.f_weight = args.r_weight, args.f_weight

    def training_step(self, batch, batch_idx):
        # infrared & visible: [b, 1, h, w]
        x, y = batch['ir'], batch['vi']

        # register (optional): infrared (fixed) & visible (moving) -> y_m (moved)
        _, y_e = canny(y)
        if self.register is not None:
            y_t = batch['vi_t']
            y_m, locs_pred, y_m_e = self.r_forward(moving=y_t, fixed=x)
        else:
            y_m, locs_pred, y_m_e = y, 0, y_e

        # fuser: infrared (infrared) & visible (visible) -> f (fusion)
        f = self.f_forward(ir=x, vi=y_m)

        # register loss (optional):
        if self.register is not None:
            # image loss: y_m_e (edges of moved) -> y (edges of visible)
            img_loss = mse_loss(y_m_e, y_e)
            self.log('reg/img', img_loss)
            # locs loss: locs_pred -> locs_gt
            locs_gt = batch['locs_gt']
            locs_loss = mse_loss(locs_pred, locs_gt)
            self.log('reg/locs', locs_loss)
            # smooth loss: y_m_e (edges of moved) smooth
            dx = torch.abs(y_m_e[:, :, 1:, :] - y_m_e[:, :, :-1, :])
            dy = torch.abs(y_m_e[:, :, :, 1:] - y_m_e[:, :, :, :-1])
            smo_loss = (torch.mean(dx * dx) + torch.mean(dy * dy)) / 2
            self.log('reg/smooth', smo_loss)
            reg_loss = img_loss * self.r_weight[0] + locs_loss * self.r_weight[1] + smo_loss * self.r_weight[2]
            self.log('train/reg', reg_loss)
        else:
            reg_loss = 0

        # fuse loss with iqa (if iqa is disabled, x_w = y_w = 1)
        x_w, y_w = batch['ir_w'], batch['vi_w']
        x_ssim = ssim_loss(f, x, window_size=11, reduction='none')
        y_ssim = ssim_loss(f, y, window_size=11, reduction='none')
        s_loss = x_ssim * x_w + y_ssim * y_w
        self.log('fus/ssim', s_loss.mean())
        x_l1 = l1_loss(f, x, reduction='none')
        y_l1 = l1_loss(f, y, reduction='none')
        l_loss = x_l1 * x_w + y_l1 * y_w
        self.log('fus/l1', l_loss.mean())
        fus_loss = self.f_weight[0] * s_loss + self.f_weight[1] * l_loss
        fus_loss = fus_loss.mean()
        self.log('train/fus', fus_loss)

        # final loss
        fin_loss = self.rf_weight[0] * reg_loss + self.rf_weight[1] * fus_loss
        self.log('train/fin', fin_loss)

        return fin_loss

    def validation_step(self, batch, batch_idx):
        # infrared & visible: [b, 1, h, w]
        x, y = batch['ir'], batch['vi']

        # output
        o = [x, y]

        # register (optional): infrared (fixed) & visible (moving) -> y_m (moved)
        if self.register is not None:
            y_t = batch['vi_t']
            y_m, _, _ = self.r_forward(moving=y_t, fixed=x)
            o += [y_t, y_m, y_m - y_t]
        else:
            y_m = y

        # fuser: ir (infrared) & vi (visible) -> f (fusion)
        f = self.f_forward(ir=x, vi=y_m)
        o += [f]

        # output
        o = torch.cat(o, dim=1)
        return o

    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> [str, Tensor]:
        # infrared & visible (moving): [b, 1, h, w]
        x, y_t = batch['ir'], batch['vi']

        # register (optional): infrared (fixed) & visible (moving) -> y_m (moved)
        if self.register is not None:
            y_m, _, _ = self.r_forward(moving=y_t, fixed=x)
        else:
            y_m = y_t

        # fuser: infrared (infrared) & visible (visible) -> f (fusion)
        f = self.f_forward(ir=x, vi=y_m)

        # output
        return batch['name'], f

    def r_forward(self, moving: Tensor, fixed: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # moving & fixed: [b, 1, h, w]
        # pred: moving & grid -> moved
        # apply transform
        moving_m, moving_e = canny(moving)
        fixed_m, fixed_e = canny(fixed)
        # predict flow
        flow = self.register(torch.cat([moving, fixed, moving_m, fixed_m], dim=1))
        flow = integrate(n_step=7, flow=flow)
        moved, locs = transformer(moving, flow)
        moved_e, locs = transformer(moving_e, flow)
        return moved, locs, moved_e

    def f_forward(self, ir: Tensor, vi: Tensor) -> Tensor:
        # ir & vi: [b, 1, h, w]
        # pred: ir (infrared) & vi (visible) -> f (fusion)
        f = self.fuser(torch.cat([ir, vi], dim=1))
        return f

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer)
        return [optimizer], {'scheduler': scheduler, 'monitor': 'train/fin'}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('ReCo+')
        # reco
        parser.add_argument('--register', type=str, default='x', help='register (m: micro, u: u-net, x: none)')
        parser.add_argument('--dim', type=int, default=32, help='dimension in backbone (default: 16)')
        # optimizer
        parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
        # weights
        parser.add_argument('--rf_weight', nargs='+', type=float, help='balance in register & fuse')
        parser.add_argument('--r_weight', nargs='+', type=float, help='balance in register: img, locs, smooth')
        parser.add_argument('--f_weight', nargs='+', type=float, help='balance in fuse: ssim, l1')

        return parent_parser
