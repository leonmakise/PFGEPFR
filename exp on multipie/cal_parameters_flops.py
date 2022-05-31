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

from eval.multadds_count import *
from network.generator import define_G

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_ids', default='1', type=str)

def main():
    global args
    args = parser.parse_args()
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    cudnn.benchmark = True

    # define generator
    netG = define_G(input_dim=3, output_dim=3)
    print('Params = {:.4f} M'.format(count_parameters_in_MB(netG)))
    print('Flops = {:.4f} M'.format(comp_multadds(netG, input_size=(3, 128, 128))))




def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters())/1e6

if __name__ == '__main__':
    main()
