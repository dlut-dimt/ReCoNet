from argparse import ArgumentParser, Namespace
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from modules.functions.integrate import integrate
from modules.functions.transformer import transformer
from modules.m_register import MRegister
from modules.u_register import URegister


class RegisterTest(pl.LightningModule):
    def __init__(self, args: Namespace):
        super().__init__()
        if args.backbone == 'm':
            print('Init with m-register')
            self.register = MRegister(in_c=2, dim=16, k_sizes=(3, 5, 7), use_bn=False, use_rs=True)
        elif args.backbone == 'u':
            print('Init with u-register')
            self.register = URegister(in_c=2, dim=16)
        else:
            assert NotImplemented, f'No match backbone: {args.backbone}'

    def training_step(self, batch, batch_idx):
        # batch: [b, 1, h, w] -> [b/2, 2, h, w] -> [b/2, (moving, fixed), h, w]
        img, _ = batch
        moving, fixed = torch.chunk(img, chunks=2, dim=0)
        # pred: moving & grid -> moved
        moved, locs = self.forward(moving, fixed)
        # loss function
        img_loss = mse_loss(moved, fixed)
        dx = torch.abs(moved[:, :, 1:, :] - moved[:, :, :-1, :])
        dy = torch.abs(moved[:, :, :, 1:] - moved[:, :, :, :-1])
        smo_loss = (torch.mean(dx * dx) + torch.mean(dy * dy)) / 2
        rig_loss = img_loss * 0.95 + smo_loss * 0.05
        return rig_loss

    def forward(self, moving: Tensor, fixed: Tensor) -> tuple[Tensor, Tensor]:
        # moving & fixed: [b, 1, h, w]
        # pred: moving & grid -> moved
        flow = self.register(torch.cat([moving, fixed], dim=1))
        flow = integrate(n_step=7, flow=flow)
        moved, locs = transformer(moving, flow)
        # output
        return moved, locs

    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> Tensor:
        # batch: [2, 1, h, w] -> [(moving, fixed), 1, h, w]
        img, _ = batch
        moving, fixed = torch.chunk(img, chunks=2, dim=0)
        # pred: moving & grid -> moved
        moved, locs = self.forward(moving, fixed)
        # output
        return torch.hstack([x.squeeze() for x in [moving, fixed, moved]])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.register.parameters(), lr=1e-3)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('RegisterTest')
        parser.add_argument('--backbone', type=str, default='m')
        return parent_parser


class SaveFigure(Callback):
    def __init__(self, dst: str | Path):
        super().__init__()
        self.dst = Path(dst)
        self.dst.mkdir(parents=True, exist_ok=True)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        rst = outputs
        save_image(rst, self.dst / f'{str(batch_idx).zfill(3)}.jpg')


def main():
    # args parser
    parser = ArgumentParser()

    # program level args
    parser.add_argument('--dst', type=str, default='tmp')
    parser.add_argument('--only_pred', action='store_true', help='use pre-trained parameters')
    parser.add_argument('--ckpt', type=str, default='', help='use your pre-trained parameters (in only_pred mode)')

    # model specific args
    parser = RegisterTest.add_model_specific_args(parser)

    # parse
    args = parser.parse_args()

    # fix seed
    pl.seed_everything(443)

    # model
    rt = RegisterTest(args)

    # dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(32, 32))])
    mnist = MNIST('./', train=True, download=True, transform=transform)

    # callbacks
    callbacks = [ModelCheckpoint(dirpath='checkpoints', every_n_train_steps=5), SaveFigure(dst=args.dst)]

    # lightning
    trainer = pl.Trainer(accelerator='gpu', devices=-1, callbacks=callbacks, max_epochs=10)

    # train
    if not args.only_pred:
        loader = DataLoader(mnist, batch_size=32, shuffle=True)
        trainer.fit(model=rt, train_dataloaders=loader)

    # predict
    loader = DataLoader(mnist, batch_size=2, shuffle=True)
    if args.only_pred:
        ckpt = f'weights/{args.backbone}-register.ckpt' if args.ckpt == '' else args.ckpt
        trainer.predict(model=rt, dataloaders=loader, ckpt_path=ckpt)
    else:
        trainer.predict(model=rt, dataloaders=loader)


if __name__ == '__main__':
    main()
