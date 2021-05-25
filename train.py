from __future__ import print_function
import sys

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from dataloader import KITTIloader2015 as lt
from dataloader import KITTILoader as DA

from models import monodepth2 as monodepth2
import random


from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torchvision.transforms as transforms


writer_train = SummaryWriter(log_dir="./logs/train")
writer_test = SummaryWriter(log_dir="./logs/test")

parser = argparse.ArgumentParser(description="FCSMNet")
parser.add_argument("--maxdisp", type=int, default=192, help="maxium disparity")
parser.add_argument("--model", default="da", help="select model")
parser.add_argument(
    "--datapath", default="/media/jiaren/ImageNet/SceneFlowData/", help="datapath"
)
parser.add_argument("--epochs", type=int, default=0, help="number of epochs to train")
parser.add_argument("--loadmodel", default=None, help="load model")
parser.add_argument("--savemodel", default="./result", help="save model")
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="enables CUDA training"
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def deterministic_training(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# seed
deterministic_training(args.seed)


(
    all_left_img,
    all_right_img,
    all_left_disp,
    test_left_img,
    test_right_img,
    test_left_disp,
) = lt.dataloader(args.datapath)


TrainImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(all_left_img, all_right_img, all_left_disp, True),
    batch_size=8,
    shuffle=True,
    num_workers=8,
    drop_last=False,
)

TestImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
    batch_size=1,
    shuffle=False,
    num_workers=1,
    drop_last=False,
)


from dataloader import KITTI_submission_loader as DA

test_left_img, test_right_img = DA.dataloader(args.datapath)


# model_encoder = monodepth2.ResnetEncoder(34, True)
# model_decoder = monodepth2.DepthDecoder(np.array([64, 64, 128, 256, 512]))

model = monodepth2.Monodepth2()

if args.cuda:
    # model_encoder = nn.DataParallel(model_encoder)
    # model_decoder = nn.DataParallel(model_decoder)
    model = nn.DataParallel(model)
    # model_encoder.cuda()
    # model_decoder.cuda()
    model.cuda()


if args.loadmodel is not None:
    print("Load pretrained model")
    pretrain_dict = torch.load(args.loadmodel)
    model.load_state_dict(pretrain_dict["state_dict"])

# print(
#     "Number of model_encoder parameters: {}".format(
#         sum([p.data.nelement() for p in model_encoder.parameters()])
#     )
# )
# print(
#     "Number of model_decoder parameters: {}".format(
#         sum([p.data.nelement() for p in model_decoder.parameters()])
#     )
# )
print(
    "Number of model_decoder parameters: {}".format(
        sum([p.data.nelement() for p in model.parameters()])
    )
)


# parameters_to_train = list(model_encoder.parameters()) + list(
#     model_decoder.parameters()
# )

# optimizer = optim.Adam(parameters_to_train, lr=0.001, betas=(0.9, 0.999))
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))


def train(imgL, imgR, disp_L, idx):
    # model_encoder.train()
    # model_decoder.train()
    model.train()

    if args.cuda:
        imgL, imgR, disp_trueL = imgL.cuda(), imgR.cuda(), disp_L.cuda()

    maskL = disp_trueL < args.maxdisp
    maskL = disp_trueL > 0  # this is required, dispity should be more than 0
    maskL.detach_()

    optimizer.zero_grad()

    start_time = time.time()

    # feature_left = model_encoder(imgL)
    # disp_left = model_decoder(feature_left)
    # disp_left = disp_left[("disp", 0)]
    disp_left = model(imgL)
    print("prediction_time = %.4f [s]" % (time.time() - start_time))

    if idx < 10:
        save_image(
            disp_left / torch.max(disp_left),
            'result/train/"left_' + test_left_img[idx].split("/")[-1],
        )

    if disp_left.ndim == 3:
        disp_left = torch.unsqueeze(disp_left, 0)
    disp_trueL = torch.unsqueeze(disp_trueL, 1)
    maskL = torch.unsqueeze(maskL, 1)

    loss = F.smooth_l1_loss(disp_left[maskL], disp_trueL[maskL], size_average=True)

    loss.backward()
    optimizer.step()

    return loss.data


def test(imgL, imgR, disp_L, idx, visualize_result=False):
    # model_encoder.eval()
    # model_decoder.eval()
    model.eval()

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

    maskL = disp_true < args.maxdisp
    maskL = disp_true > 0
    maskL.detach_()

    if imgL.shape[2] % 32 != 0:
        times = imgL.shape[2] // 32
        top_pad = (times + 1) * 32 - imgL.shape[2]
    else:
        top_pad = 0

    if imgL.shape[3] % 32 != 0:
        times = imgL.shape[3] // 32
        right_pad = (times + 1) * 32 - imgL.shape[3]
    else:
        right_pad = 0

    imgL = F.pad(imgL, (0, right_pad, top_pad, 0))
    imgR = F.pad(imgR, (0, right_pad, top_pad, 0))

    with torch.no_grad():
        start_time = time.time()
        # feature_left = model_encoder(imgL)
        # disp_left = model_decoder(feature_left)
        # disp_left = disp_left[("disp", 0)]
        disp_left = model(imgL)
        # print('prediction_time = %.4f [s]' %(time.time() - start_time))

    if idx < 10:
        save_image(
            disp_left / torch.max(disp_left),
            'result/train/"left_' + test_left_img[idx].split("/")[-1],
        )

    if disp_left.ndim == 4:
        disp_left = torch.squeeze(disp_left, 0)

    if top_pad != 0 or right_pad != 0:
        img = disp_left[:, top_pad:, :-right_pad]
    else:
        img = disp_left

    if len(disp_true[maskL]) == 0:
        loss = 0
    else:
        loss = F.l1_loss(img[maskL], disp_true[maskL])
        # torch.mean(torch.abs(img[mask]-disp_true[mask]))  # end-point-error

    return loss.data.cpu()


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001
    if epoch < 100:
        lr = 0.001
    elif epoch < 200:
        lr = 0.0005
    else:
        lr = 0.0001

    print("learning rate = ", lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def main():
    start_full_time = time.time()
    iteration = 0
    for epoch in range(0, args.epochs):
        # adjust_learning_rate(optimizer, epoch)

        ## training ##
        total_train_loss = 0
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
            start_time = time.time()
            loss = train(imgL_crop, imgR_crop, disp_crop_L, batch_idx)
            iteration += 1
            print(
                "epoch %d , Iter %d , traing_time = %.4f"
                % (epoch, batch_idx, time.time() - start_time)
            )
            total_train_loss += loss
        avg_train_loss = total_train_loss / len(TrainImgLoader)
        print("epoch %d , avg_train_loss = %.4f" % (epoch, avg_train_loss))
        writer_train.add_scalar("avg_train_loss", avg_train_loss, epoch)

        # SAVE
        savefilename = "result/weights" + "/" + args.model + str(epoch) + ".tar"
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "train_loss": total_train_loss / len(TrainImgLoader),
            },
            savefilename,
        )

        # test
        total_test_loss = 0
        for batch_idx, (imgL_crop_test, imgR_crop_test, disp_crop_L_test) in enumerate(
            TestImgLoader
        ):
            total_test_loss += test(
                imgL_crop_test, imgR_crop_test, disp_crop_L_test, batch_idx
            )

        avg_test_loss = total_test_loss / len(TestImgLoader)
        writer_test.add_scalar("avg_test_loss", avg_test_loss, epoch)
        print("epoch %d , avg_test_loss = %.4f " % (epoch, avg_test_loss))

    writer_train.close()
    writer_test.close()


if __name__ == "__main__":
    main()
