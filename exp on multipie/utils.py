import os
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F


# L2 normalization
def norm(x):
    x = F.normalize(x, p=2, dim=1)
    return x

# cosine similarity
def angle(x, y):
    angle = torch.abs((x * y).sum(dim=1)).sum()
    angle = angle / float(x.size(0))
    return angle

# convert rgb to gray
def rgb2gray(img):
    r, g, b = torch.split(img, 1, dim=1)
    return torch.mul(r, 0.299) + torch.mul(g, 0.587) + torch.mul(b, 0.114)

# load pretrained model
def load_model(model, pretrained):
    weights = torch.load(pretrained)
    pretrained_dict = weights['model'].state_dict()
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

# save model
def save_checkpoint(model, epoch, iteration, name):
    model_out_path = "model/" + name + "model_epoch_{}_iter_{}.pth".format(epoch, iteration)
    state = {"epoch": epoch, "model": model}
    if not os.path.exists("model/"):
        os.makedirs("model/")
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
