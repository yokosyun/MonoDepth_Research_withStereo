import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np
from . import preprocess
import torch.nn as nn

from torchvision.utils import save_image

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert("RGB")


def disparity_loader(path):
    return Image.open(path)


class myImageFloder(data.Dataset):
    def __init__(
        self,
        left,
        right,
        left_disparity,
        training,
        loader=default_loader,
        dploader=disparity_loader,
    ):

        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL = self.dploader(disp_L)

        if self.training:
            w, h = left_img.size
            # th, tw = 368, 1232
            th, tw = 256, 512

            if True:
                x1 = np.random.randint(0, w - tw)
                y1 = np.random.randint(0, h - th)

                left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
                right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

                dataL = np.ascontiguousarray(dataL, dtype=np.float32) / 256
                dataL = dataL[y1 : y1 + th, x1 : x1 + tw]
            else:
                width, height = left_img.width, left_img.height

                scale = np.random.randint(1, 3) + np.random.rand()
                w = int(width * scale)
                h = int(height * scale)

                left_img = left_img.resize((w, h))
                right_img = right_img.resize((w, h))
                dataL = dataL.resize((w, h))

                x1 = np.random.randint(0, w - tw)
                y1 = np.random.randint(0, h - th)

                left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
                right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

                dataL = np.ascontiguousarray(dataL, dtype=np.float32) / 256
                dataL = dataL[y1 : y1 + th, x1 : x1 + tw]
                dataL *= scale

            processed = preprocess.get_transform(augment=False)
            left_img = transforms.ToTensor()(left_img)
            right_img = transforms.ToTensor()(right_img)

            return left_img, right_img, dataL
        else:
            w, h = left_img.size

            left_img = left_img.crop((w - 1232, h - 368, w, h))
            right_img = right_img.crop((w - 1232, h - 368, w, h))
            w1, h1 = left_img.size

            dataL = dataL.crop((w - 1232, h - 368, w, h))
            dataL = np.ascontiguousarray(dataL, dtype=np.float32) / 256

            processed = preprocess.get_transform(augment=False)

            left_img = transforms.ToTensor()(left_img)
            right_img = transforms.ToTensor()(right_img)

            return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)
