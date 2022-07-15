from functools import reduce
from pathlib import Path
from typing import Literal, List

import cv2
import torch
from kornia import image_to_tensor, create_meshgrid
from torch import Tensor, nn
from torch.utils.data import Dataset
from torchvision.transforms import Resize, RandomResizedCrop

from modules.random_adjust import RandomAdjust


class AutoRF(Dataset):
    def __init__(
            self,
            root: str | Path,
            mode: str = Literal['train', 'val', 'pred'],
            level: str = Literal['none', 'easy', 'normal', 'hard'],
            iqa: bool = True,
    ):
        super().__init__()
        root = Path(root)
        self.root = root
        self.mode = mode
        self.iqa = iqa

        # filter
        list_f = root / 'meta' / f'{mode}.txt'
        assert list_f.exists(), f'find no meta file in path {str(list_f)}'
        t = list_f.read_text().splitlines()

        # get sample list
        types = ['.jpg', '.bmp', '.png']
        samples = [x.name for x in sorted((root / 'ir').glob('*')) if x.suffix in types]
        samples = list(filter(lambda x: Path(x).stem in t, samples))
        self.samples = samples

        # init complex transform
        match level:
            case 'easy':
                easy = {'transforms': 'ep', 'kernel_size': (143, 143), 'sigma': (32, 32), 'distortion_scale': 0.2}
                self.adjust = RandomAdjust(easy)
            case 'normal':
                normal = {'transforms': 'ep', 'kernel_size': (103, 103), 'sigma': (32, 32), 'distortion_scale': 0.3}
                self.adjust = RandomAdjust(normal)
            case 'hard':
                hard = {'transforms': 'ep', 'kernel_size': (63, 63), 'sigma': (32, 32), 'distortion_scale': 0.4}
                self.adjust = RandomAdjust(hard)
            case _:
                self.adjust = None

        # init transform
        match mode:
            case 'train':
                self.transform = RandomResizedCrop(size=(320, 320))
            case 'val':
                self.transform = Resize(size=(320, 320))
            case _:
                self.transform = nn.Identity()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        # find sample with index
        name = self.samples[index]

        # load infrared and visible
        x = self.gray_read(self.root / 'ir' / name)
        y = self.gray_read(self.root / 'vi' / name)

        # load information measurement
        if self.iqa:
            x_w = self.gray_read(self.root / 'iqa' / 'ir' / name)
            y_w = self.gray_read(self.root / 'iqa' / 'vi' / name)
        else:
            x_w, y_w = torch.ones_like(x), torch.ones_like(x)

        # transform (resize)
        t = torch.cat([x, y, x_w, y_w], dim=0)
        x, y, x_w, y_w = torch.chunk(self.transform(t), chunks=4, dim=0)

        # adjust (challenge simulations) (optional)
        h, w = y.size()[-2:]
        grid = create_meshgrid(h, w, device=y.device).to(y.dtype)
        if self.adjust is not None:
            y_t, params = self.adjust(y.unsqueeze(dim=0))
            flow_gt = reduce(lambda i, j: i + j, [v for _, v in params.items()])
            locs_gt = grid - flow_gt
            y_t.squeeze_(dim=0)  # [1, 1, h, w] -> [1, h, w]
        else:
            y_t = y
            locs_gt = grid
        locs_gt.squeeze_(dim=0)  # [1, h, w, 2] -> [h, w, 2]

        # merge data
        sample = {'name': name, 'ir': x, 'vi': y, 'ir_w': x_w, 'vi_w': y_w, 'vi_t': y_t, 'locs_gt': locs_gt}

        # return as except
        return sample

    @staticmethod
    def gray_read(img_path: str | Path) -> Tensor:
        img_n = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img_t = image_to_tensor(img_n).float() / 255
        return img_t

    @staticmethod
    def collate_fn(data: List[dict]) -> dict:
        # keys
        keys = data[0].keys()
        # merge
        new_data = {}
        for key in keys:
            k_data = [d[key] for d in data]
            new_data[key] = k_data if isinstance(k_data[0], str) else torch.stack(k_data)
        # return as expected
        return new_data
