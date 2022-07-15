import argparse
import pathlib
import statistics
import time

import cv2
import kornia
import torch
import torch.backends.cudnn
from tqdm import tqdm

from archive.model import Fuser


class Fuse:
    """
    Fuse images with given args.
    """

    def __init__(self, checkpoint: pathlib.Path, loop_num: int = 3, dim: int = 64):
        """
        Init model and load pre-trained parameters.
        :param checkpoint: pre-trained model checkpoint
        :param loop_num: AFuse recurrent loop number, default: 3
        :param dim: AFuse feather number, default: 64
        """

        # device [cuda or cpu]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        # load pre-trained network
        net = Fuser(loop_num=loop_num, feather_num=dim)
        net.load_state_dict(torch.load(str(checkpoint), map_location='cpu'))
        net.to(device)
        net.eval()
        self.net = net

    @torch.no_grad()
    def __call__(self, ir_path: pathlib.Path, vi_path: pathlib.Path, dst: pathlib.Path):
        """
        Fuse image with infrared folder, visible folder and destination path.
        :param ir_path: infrared folder path
        :param vi_path: visible folder path
        :param dst: fused images destination path
        """

        # src list
        ir_list = [x for x in ir_path.glob('*') if x.suffix in ['.bmp', '.jpg', '.png']]
        vi_list = [x for x in vi_path.glob('*') if x.suffix in ['.bmp', '.jpg', '.png']]

        # time record
        fuse_time = []

        # fuse images
        src = tqdm(zip(ir_list, vi_list))
        for ir_path, vi_path in src:
            "fuse one pair with src image path"

            # judge image pair
            assert ir_path.name == vi_path.name
            src.set_description(f'fuse {ir_path.name}')

            # read image with Tensor
            ir = self._imread(ir_path).unsqueeze(0)
            vi = self._imread(vi_path).unsqueeze(0)
            ir = ir.to(self.device)
            vi = vi.to(self.device)

            # network flow
            torch.cuda.synchronize() if str(self.device) == 'cuda' else None
            start = time.time()
            im_f, _, _ = self.net([ir, vi])
            torch.cuda.synchronize() if str(self.device) == 'cuda' else None
            end = time.time()
            fuse_time.append(end - start)

            # save fusion image
            self._imsave(dst / ir_path.name, im_f[-1])

        # analyze fuse time
        std = statistics.stdev(fuse_time[1:])
        avg = statistics.mean(fuse_time[1:])
        print(f'fuse std time: {std:.4f}(s)')
        print(f'fuse avg time: {avg:.4f}(s)')
        print('fps (equivalence): {:.4f}'.format(1. / avg))

    @staticmethod
    def _imread(path: pathlib.Path, flags=cv2.IMREAD_GRAYSCALE) -> torch.Tensor:
        im_cv = cv2.imread(str(path), flags)
        im_ts = kornia.utils.image_to_tensor(im_cv / 255.0).type(torch.FloatTensor)
        return im_ts

    @staticmethod
    def _imsave(path: pathlib.Path, image: torch.Tensor):
        im_ts = image.squeeze().cpu()
        path.parent.mkdir(parents=True, exist_ok=True)
        im_cv = kornia.utils.tensor_to_image(im_ts) * 255.
        cv2.imwrite(str(path), im_cv)


def hyper_args():
    """
    get hyper parameters from args
    """

    parser = argparse.ArgumentParser(description='ReCo(v0) fuse process')

    # dataset
    parser.add_argument('--ir', default='../data/tno/ir', help='infrared image folder')
    parser.add_argument('--vi', default='../data/tno/vi', help='visible image folder')
    parser.add_argument('--dst', default='../runs/archive', help='fuse image save folder')
    # checkpoint
    parser.add_argument('--cp', default='params.pth', help='weight checkpoint')
    # fuse network
    parser.add_argument('--loop', default=3, type=int, help='fuse loop time')
    parser.add_argument('--dim', default=64, type=int, help='fuse feather dim')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # hyper parameters
    args = hyper_args()

    f = Fuse(checkpoint=pathlib.Path(args.cp), loop_num=args.loop, dim=args.dim)
    f(pathlib.Path(args.ir), pathlib.Path(args.vi), pathlib.Path(args.dst))
