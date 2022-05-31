import numpy as np
import os, random, copy
from PIL import Image
from collections import defaultdict

import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class MutiPIE_Dataset(data.Dataset):
    def __init__(self, args):
        super(MutiPIE_Dataset, self).__init__()
        
        self.img_root = args.img_root
        self.listfile = args.train_list

        self.transform = transforms.Compose([
            transforms.CenterCrop(128),
            transforms.ToTensor()
        ])

        self.img_list, self.makePairDict = self.file_reader()

    def __getitem__(self, index):

        img_name = self.img_list[index]
        pid, sid, eid, cid, iid, _ = img_name.strip().split('_')
        img_path = os.path.join(self.img_root, img_name)

        # randomly select two frontal images: img_pair_a and img_pair_b
        img_pair_a_name, img_pair_b_name = self.get_pair(pid)
        img_pair_a_path = os.path.join(self.img_root, img_pair_a_name)
        img_pair_b_path = os.path.join(self.img_root, img_pair_b_name)

        #
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        img_pair_a = Image.open(img_pair_a_path).convert('RGB')
        img_pair_a = self.transform(img_pair_a)

        img_pair_b = Image.open(img_pair_b_path).convert('RGB')
        img_pair_b = self.transform(img_pair_b)

        return {'img': img, 'img_pair_a': img_pair_a, 'img_pair_b': img_pair_b, 'label': (int(pid)-1)}

    def __len__(self):
        return len(self.img_list)

    def file_reader(self):
        def dict_profile():
            return {'front': [], 'profile': []}

        img_list = []
        makePairDict = defaultdict(dict_profile)
        with open(self.listfile) as f:
            img_names = f.readlines()
            for img_name in img_names:
                img_name = img_name.strip()
                pid, sid, eid, cid, iid, _ = img_name.split('_') # pid denotes the identity

                img_list.append(img_name)

                if cid == '051': # cid == '051' means it is a frontal image
                    makePairDict[pid]['front'].append(img_name)
                else:
                    makePairDict[pid]['profile'].append(img_name)

        return img_list, makePairDict

    def get_pair(self, pid):
        img_pair = self.makePairDict[pid]['front']
        if len(img_pair) >= 2:
            img_pair = random.sample(img_pair, 2)
            return img_pair[0], img_pair[1]
        else:
            return img_pair[0], img_pair[0]
