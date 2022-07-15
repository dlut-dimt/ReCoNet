from pathlib import Path

import cv2
import torch
from kornia import image_to_tensor
from kornia.filters import canny
from torch import Tensor
from torchvision.transforms import Normalize
from torchvision.utils import save_image


def gray_read(img_path: str | Path) -> Tensor:
    img_n = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    img_t = image_to_tensor(img_n).float() / 255
    return img_t


def find_adjust():
    # init normalize
    norm = Normalize(mean=0.44, std=0.27)
    # load source image
    ir = gray_read('../../data/tno/ir/A_028.bmp')
    vi = gray_read('../../data/tno/vi/A_028.bmp')
    # apply canny filter
    ir_mag, ir_e = canny(ir.unsqueeze(0))
    vi_mag, vi_e = canny(vi.unsqueeze(0))
    # magnitude max
    ir_max = torch.where(ir_mag > 0.1, 1, 0)
    vi_max = torch.where(vi_mag > 0.1, 1, 0)
    # output
    img = torch.hstack([x.squeeze() for x in [ir_mag, ir_e, ir_max, vi_mag, vi_e, vi_max]])
    save_image(img, 'tmp.jpg')


if __name__ == '__main__':
    find_adjust()
