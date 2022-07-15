from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor
from torch.utils.data import DataLoader

from lightning.auto_rf import AutoRF
from lightning.reco import ReCo
from utils.pretty_vars import pretty_vars


class LogImageCallback(Callback):
    def __init__(self, logger: WandbLogger, show_grad: bool = False):
        super().__init__()
        self.logger = logger
        self.show_grad = show_grad

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx == 0:
            assert isinstance(outputs, Tensor)
            size = outputs.shape[1]
            imgs = [torch.clip(img[-1].squeeze(), min=0, max=1) for img in torch.chunk(outputs, chunks=size, dim=1)]
            captions = ['ir', 'vi', 'vi_t', 'vi_m', 'dif', 'f'] if size == 6 else ['ir', 'vi', 'f']
            self.logger.log_image(key='sample', images=imgs, caption=captions)

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        if self.show_grad:
            print(pretty_vars(pl_module.register))


def main():
    # args parser
    parser = ArgumentParser()

    # program level args
    # lightning
    parser.add_argument('--ckpt', type=str, default='checkpoints', help='checkpoints save folder')
    parser.add_argument('--show_grad', action='store_true', help='show grad before zero_grad')
    parser.add_argument('--seed', type=int, default=443, help='seed for random number')
    # wandb
    parser.add_argument('--key', type=str, help='wandb auth key')
    # auto rf
    parser.add_argument('--data', type=str, default='../data/tno', help='input data folder')
    parser.add_argument('--deform', type=str, default='none', help='random adjust level')
    # loader
    parser.add_argument('--bs', type=int, default=32, help='batch size')
    # cuda
    parser.add_argument('--no_cuda', action='store_true', help='disable cuda (for cpu and out of memory)')

    # model specific args
    parser = ReCo.add_model_specific_args(parser)

    # parse
    args = parser.parse_args()

    # fix seed
    pl.seed_everything(args.seed)

    # model
    reco = ReCo(args)

    # dataloader
    train_dataset = AutoRF(root=args.data, mode='train', level=args.deform)
    train_loader = DataLoader(
        train_dataset, batch_size=args.bs,
        shuffle=True, collate_fn=AutoRF.collate_fn, num_workers=72,
    )
    val_dataset = AutoRF(root=args.data, mode='val', level=args.deform)
    val_loader = DataLoader(
        val_dataset, batch_size=1,
        collate_fn=AutoRF.collate_fn, num_workers=72,
    )

    # logger
    wandb.login(key=args.key)
    logger = WandbLogger(project='reco')

    # callbacks
    callbacks = [
        ModelCheckpoint(dirpath=args.ckpt, every_n_train_steps=10),
        LogImageCallback(logger=logger, show_grad=args.show_grad),
        LearningRateMonitor(logging_interval='step'),
    ]

    # lightning
    accelerator, devices, strategy = ('cpu', None, None) if args.no_cuda else ('gpu', -1, 'ddp')
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        callbacks=callbacks,
        max_epochs=800,
        strategy=strategy,
        log_every_n_steps=5,
    )
    trainer.fit(model=reco, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()
