import os
import numpy as np
import math
import argparse
import collections
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser()
parser.add_argument('--root_feat', default='../feat', type=str)
parser.add_argument('--probe_list_path', default='./', type=str)
parser.add_argument('--gallery_list_path', default='./', type=str)


def read_img_list(img_list_path):
    img_list = []
    labels = []
    with open(img_list_path, 'r') as file:
        for line in file.readlines():
            tmp = line.strip().split(' ')
            img_list.append(tmp[0])
            labels.append(int(tmp[1]))
    return img_list, labels

def load_feat(root_feat, img_list):
    all_feat = []
    for feat_name in tqdm(img_list):
        feat_path = os.path.join(root_feat, feat_name)
        feat_path = os.path.splitext(feat_path)[0] + '.feat'
        feat = np.fromfile(feat_path, dtype=np.float32)
        all_feat.append(feat)
    all_feat = np.vstack(all_feat)
    return all_feat

def get_1n_scores_and_label(gallery_feat, probe_feat, gallery_label, probe_label, g_p_issame=False):
    score_matrix = cosine_similarity(gallery_feat, probe_feat)

    num_g = len(gallery_label)
    num_p = len(probe_label)

    gallery_label = np.array(gallery_label).reshape((num_g, 1))
    probe_label = np.array(probe_label).reshape((num_p, 1))

    label_matrix = get_label_matrix(gallery_label, probe_label)
    label_matrix = label_matrix.astype(np.int32)

    if g_p_issame:
        np.fill_diagonal(score_matrix, 0)
        np.fill_diagonal(label_matrix, 0)

    return score_matrix, label_matrix

def get_label_matrix(gallery_label, probe_label):
    num_g = gallery_label.shape[0]
    num_p = probe_label.shape[0]
    label_matrix = np.zeros((num_g, num_p))

    g_l = np.broadcast_to(gallery_label, (num_g, num_p))
    p_l = np.broadcast_to(probe_label.T, (num_g, num_p))
    label_matrix[np.where(g_l == p_l)] = 1
    return label_matrix

def get_rank_and_hit(score_matrix, label_matrix, ranks, g_p_issame=False):
    score_matrix = torch.from_numpy(score_matrix)
    label_matrix = torch.from_numpy(label_matrix)
    _, pred = torch.topk(score_matrix, max(ranks), 0, True, True)
    label_max = torch.gather(label_matrix, 0, pred)

    hit_list = list()

    total_num_of_hits = torch.sum(torch.sum(label_matrix, 0) > 0)

    if g_p_issame == True:
        total_num_of_hits = total_num_of_hits - 1

    for r in ranks:
        l = label_max[:r, :]
        l = torch.sum(l, dim=0) > 0
        hit = torch.sum(l)
        hit_list.append((hit * 1.0).type(torch.FloatTensor) / total_num_of_hits.float())
    return hit_list

def evaluation_1n(root_feat, probe_list_path, gallery_list_path, g_p_issame=False, ranks=[1, 3, 5, 10, 20]):
    probe_list, probe_label = read_img_list(probe_list_path)
    gallery_list, gallery_label = read_img_list(gallery_list_path)

    probe_feat = load_feat(root_feat, probe_list)
    gallery_feat = load_feat(root_feat, gallery_list)
    score_matrix, label_matrix = get_1n_scores_and_label(gallery_feat, probe_feat, gallery_label, probe_label,
                                                         g_p_issame=g_p_issame)
    hit_lst = get_rank_and_hit(score_matrix, label_matrix, ranks, g_p_issame=g_p_issame)
    for i, hit in enumerate(hit_lst):
        print('Rank-{}:\t{}'.format(ranks[i], hit))


if __name__ == '__main__':
    args = parser.parse_args()

    root_feat = args.root_feat
    probe_list_path = args.probe_list_path
    gallery_list_path = args.gallery_list_path

    evaluation_1n(root_feat, probe_list_path, gallery_list_path)
