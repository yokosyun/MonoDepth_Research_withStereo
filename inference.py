from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
import math
from models import monodepth2 as monodepth2

from PIL import Image
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description="FCSMNet")
parser.add_argument("--KITTI", default="2015", help="KITTI version")
parser.add_argument(
    "--datapath",
    default="/media/yoko/SSD-PGU3/workspace/datasets/KITTI/data_scene_flow/training/",
    help="select model",
)
parser.add_argument("--loadmodel", default="./result/model.tar", help="loading model")
parser.add_argument("--model", default="FCSMNet", help="select model")
parser.add_argument("--maxdisp", type=int, default=192, help="maxium disparity")
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="enables CUDA training"
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)

args = parser.parse_args()
if args.KITTI == "2015":
    from dataloader import KITTI_submission_loader as DA
else:
    from dataloader import KITTI_submission_loader2012 as DA

test_left_img, test_right_img = DA.dataloader(args.datapath)


args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


model = monodepth2.Monodepth2()

model = nn.DataParallel(model, device_ids=[0])
if args.cuda:
    model.cuda()


if args.loadmodel is not None:
    print("load model")
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict["state_dict"])

print(
    "Number of model parameters: {}".format(
        sum([p.data.nelement() for p in model.parameters()])
    )
)


print("args.cuda=", args.cuda)


def test(imgL, imgR):
    model.eval()

    if args.cuda:
        imgL = imgL.cuda()
        imgR = imgR.cuda()

    with torch.no_grad():
        start_time = time.time()
        pred_dispL = model(imgL)
        processing_time = time.time() - start_time
        print("time = %.4f" % (processing_time))

    pred_dispL = torch.squeeze(pred_dispL)
    pred_dispL = pred_dispL.data.cpu().numpy()

    return pred_dispL, processing_time


def main():
    total_time = 0
    cnt = 0

    for inx in range(len(test_left_img)):

        imgL_o = Image.open(test_left_img[inx]).convert("RGB")
        imgR_o = Image.open(test_right_img[inx]).convert("RGB")

        imgL = transforms.ToTensor()(imgL_o)
        imgR = transforms.ToTensor()(imgR_o)

        # pad to width and hight to 32 times
        if imgL.shape[1] % 32 != 0:
            times = imgL.shape[1] // 32
            top_pad = (times + 1) * 32 - imgL.shape[1]
        else:
            top_pad = 0

        if imgL.shape[2] % 32 != 0:
            times = imgL.shape[2] // 32
            right_pad = (times + 1) * 32 - imgL.shape[2]
        else:
            right_pad = 0

        imgL = F.pad(imgL, (0, right_pad, top_pad, 0)).unsqueeze(0)
        imgR = F.pad(imgR, (0, right_pad, top_pad, 0)).unsqueeze(0)

        pred_dispL, processing_time = test(imgL, imgR)

        if inx >= 10:
            total_time += processing_time
            cnt += 1

        if top_pad != 0 or right_pad != 0:
            img = pred_dispL[top_pad:, :-right_pad]
        else:
            img = pred_dispL

        # save image
        if True:
            img = (img * 256).astype("uint16")
            img = Image.fromarray(img)
            img.save("result/inference/" + test_left_img[inx].split("/")[-1])

    print("average processing time = ", total_time / cnt)


if __name__ == "__main__":
    main()
