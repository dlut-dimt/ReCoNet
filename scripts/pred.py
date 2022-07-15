from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import Callback
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from lightning.auto_rf import AutoRF
from lightning.reco import ReCo


class SaveFigure(Callback):
    def __init__(self, dst: str | Path):
        super().__init__()
        self.dst = Path(dst)
        self.dst.mkdir(parents=True, exist_ok=True)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        name, f = outputs
        save_image(f.squeeze(), self.dst / name[0])


def main():
    # args parser
    parser = ArgumentParser()

    # program level args
    # lightning
    parser.add_argument('--ckpt', type=str, default='../weights/default-f.ckpt', help='checkpoint path')
    # auto rf
    parser.add_argument('--data', type=str, default='../data/tno', help='input data folder')
    parser.add_argument('--deform', type=str, default='none', help='random adjust level')
    # reco
    parser.add_argument('--dst', type=str, default='runs', help='output save folder')
    # cuda
    parser.add_argument('--no_cuda', action='store_true', help='disable cuda (for cpu and out of memory)')

    # model specific args
    parser = ReCo.add_model_specific_args(parser)

    # parse
    args = parser.parse_args()

    # fix seed
    pl.seed_everything(443)

    # model
    reco = ReCo(args)

    # dataloader
    dataset = AutoRF(root=args.data, mode='pred', level=args.deform)
    loader = DataLoader(dataset, collate_fn=AutoRF.collate_fn)

    # callbacks
    callbacks = [SaveFigure(dst=args.dst)]

    # lightning
    accelerator, devices, strategy = ('cpu', None, None) if args.no_cuda else ('gpu', -1, 'ddp')
    trainer = pl.Trainer(accelerator=accelerator, devices=devices, callbacks=callbacks, strategy=strategy)
    trainer.predict(model=reco, dataloaders=loader, ckpt_path=args.ckpt)


if __name__ == '__main__':
    main()
