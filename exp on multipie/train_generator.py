import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import torchvision.transforms as transforms

from utils import *
from data.dataset_multipie import MutiPIE_Dataset
from network.light_cnn import define_LightCNN
from network.generator import define_G


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_ids', default='1,3,6,7', type=str)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--epochs', default=80, type=int)
parser.add_argument('--outf', default='results/', type=str, help='save the output image')

parser.add_argument('--pre_epoch', default=0, type=int, help='restart from previous model')
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--lr', default=0.0002, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=1e-4)
parser.add_argument('--print_freq', default=1, type=int, help='print log')

parser.add_argument('--weights', default='./pre_train/lightCNN_152_checkpoint.pth.tar', type=str, help='the weight of lightcnn pretrained on Multi-PIE')
parser.add_argument('--img_root', default='../data/Multi-PIE/FS_aligned', type=str)
parser.add_argument('--train_list', default='../data/Multi-PIE/FS/train.csv', type=str)


def main():
    global args
    args = parser.parse_args()
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    cudnn.benchmark = True

    if not os.path.exists(args.outf):
        os.makedirs(args.outf)

    # define lightcnn
    LightCNN = define_LightCNN()
    # load pretrained lightcnn
    print("=> loading pretrained lightcnn model '{}'".format(args.weights))
    checkpoint = torch.load(args.weights)
    pretrained_dict = checkpoint['state_dict']
    model_dict = LightCNN.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    LightCNN.load_state_dict(model_dict)

    for param in LightCNN.parameters():
        param.requires_grad = False

    LightCNN.eval()

    # define generator
    netG = define_G(input_dim=3, output_dim=3)

    # load pretrained model
    if args.pre_epoch:
        print('load pretrained model %d' % args.pre_epoch)
        load_model(netG, './model/netG_model_epoch_%d_iter_0.pth' % args.pre_epoch)

    # dataset
    train_loader = torch.utils.data.DataLoader(
        MutiPIE_Dataset(args), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    # optimizer
    optimizer = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # criterion
    criterionMSE = nn.MSELoss().cuda()
    criterionPix = torch.nn.L1Loss().cuda()

    # training phase
    netG.train()
    start_epoch = args.pre_epoch + 1
    for epoch in range(start_epoch, args.epochs + 1):

        for i, data in enumerate(train_loader):
            # load data
            img = Variable(data['img'].cuda()) #img是个侧脸
            img_pair_a = Variable(data['img_pair_a'].cuda())
            img_pair_b = Variable(data['img_pair_b'].cuda())#img_pair_a,b是和img同个人的随机两张正脸

            # extract identity features
            id = LightCNN(rgb2gray(img)) #id是正脸id
            id_pair_a = LightCNN(rgb2gray(img_pair_a)) #id_pair_a,b是两个侧脸id
            id_pair_b = LightCNN(rgb2gray(img_pair_b))

            # forward
            dx = netG(img)
            fake = img - dx#fake是重建的侧脸

            id_fake = LightCNN(rgb2gray(fake))#id_fake是重建侧脸的id

            loss_s = (criterionMSE(id_fake, id_pair_a) + criterionMSE(id_fake, id_pair_b)) / 2.0 #loss_s就是重建侧脸id与两个正脸id之间差距的平均值
            loss_pix = criterionPix(fake, img) #loss_pix是重建侧脸与原侧脸图之间的像素级差距

            loss = loss_pix + 25 * loss_s
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % args.print_freq == 0:
                # observe the training process
                angle_pair = (angle(id_fake, id_pair_a) + angle(id_fake, id_pair_b)) / 2.0
                angle_pair_real = (angle(id, id_pair_a) + angle(id, id_pair_b)) / 2.0
                angle_self = angle(id_fake, id)

                info = '====> Epoch: [{:0>3d}][{:3d}/{:3d}] | '.format(epoch, i, len(train_loader))
                info += 'Angle: {:4.3f} ({:4.3f}) ({:4.3f}) | Loss: pix: {:4.3f} s: {:4.3f} | '.format(
                    angle_pair.item(), angle_pair_real.item(), angle_self.item(), loss_pix.item(), loss_s.item())

                print(info)

            if (i != 0) and (i % 200 == 0):
                vutils.save_image(torch.cat([img_pair_a[0:24], img_pair_b[0:24], img[0:24], fake[0:24], dx[0:24]], dim=0).data,
                                  '{}/Epoch_{:03d}_Iter_{:06d}_img.png'.format(args.outf, epoch, i), nrow=24)

        # save model
        save_checkpoint(netG, epoch, 0, 'netG_')


if __name__ == '__main__':
    main()
