import os
import time
import cv2
import argparse
import numpy as np
from PIL import Image
from collections import OrderedDict

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

from utils import *
from network.light_cnn import define_LightCNN
from network.generator import define_G

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_ids', default='0', type=str)
parser.add_argument('--weights', default='', type=str)
parser.add_argument('--root_path', default='', type=str)
parser.add_argument('--img_list', default='', type=str)
parser.add_argument('--save_path', default='', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--generation', default=1, type=int)


def default_loader(path):
    print(path)
    img = Image.open(path).convert('RGB') #.convert('L')
    return img

def default_list_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgPath, label = line.strip().split(' ')
            imgList.append((imgPath, int(label)))
    return imgList

class ImageList(data.Dataset):
    def __init__(self, root, fileList, transform=None, list_reader=default_list_reader, loader=default_loader):
        self.root = root
        self.imgList = list_reader(fileList)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_name, target = self.imgList[index]
        img = self.loader(os.path.join(self.root, img_name))
        print(img)

        if self.transform is not None:
            img = self.transform(img)

        return img_name, img, target

    def __len__(self):
        return len(self.imgList)


def main():
    global args
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    cudnn.benchmark = True

    # define lightcnn
    LightCNN = define_LightCNN()
    # load pretrained lightcnn
    checkpoint = torch.load("./pre_train/LightCNN_29Layers_V2_checkpoint.pth.tar")
    pretrained_dict = checkpoint['state_dict']
    model_dict = LightCNN.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    LightCNN.load_state_dict(model_dict)

    for param in LightCNN.parameters():
        param.requires_grad = False

    LightCNN.eval()

    # generator
    netG = define_G(input_dim=3, output_dim=3)
    load_model(netG, args.weights)

    for param in netG.parameters():
        param.requires_grad = False

    netG.eval

    #
    img_list = read_list(args.img_list)
    dataset = torch.utils.data.DataLoader(
                ImageList(root=args.root_path, fileList=args.img_list,
                    transform=transforms.Compose([
                        transforms.CenterCrop(128),
                        #transforms.Resize(128),
                        transforms.ToTensor()
                    ])),
                batch_size=args.batch_size, shuffle=True,
                num_workers=0, pin_memory=True)

    count = 0
    end = time.time()
    for i, (img_names, input, label) in enumerate(dataset):
        print(img_names)

        start = time.time()
        input = Variable(input.cuda())

        if args.generation:
            with torch.no_grad():
                dx = netG(input)
                fake = input - dx
                features = LightCNN(rgb2gray(fake))

            end = time.time() - start

            for j, img_name in enumerate(img_names):
                count = count + 1
                feat = features[j, :]
                print("{}({}/{}). Time: {}".format(os.path.join(args.root_path, img_name), count, len(img_list),
                                                   end / int(args.batch_size)))
                save_feature(args.save_path, img_name, feat.data.cpu().numpy())
        else:
            with torch.no_grad():
                features = LightCNN(rgb2gray(input))

            end = time.time() - start

            for j, img_name in enumerate(img_names):
                count = count + 1
                feat = features[j, :]
                print("{}({}/{}). Time: {}".format(os.path.join(args.root_path, img_name), count, len(img_list),
                                                   end / int(args.batch_size)))
                save_feature(args.save_path, img_name, feat.data.cpu().numpy())


def read_list(list_path):
    img_list = []
    with open(list_path, 'r') as f:
        for line in f.readlines()[0:]:
            img_path = line.strip().split()
            img_list.append(img_path[0])
    print('There are {} images..'.format(len(img_list)))
    return img_list

def save_feature(save_path, img_name, features):
    img_path = os.path.join(save_path, img_name)
    img_dir = os.path.dirname(img_path) + '/';
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    fname = os.path.splitext(img_path)[0]
    fname = fname + '.feat'
    fid = open(fname, 'wb')
    fid.write(features)
    fid.close()

if __name__ == '__main__':
    main()
