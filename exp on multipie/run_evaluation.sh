#!/bin/bash

echo extract features

epoch_number=$1

gpu_ids="0"
batch_size=128
save_path="./feat"
weights="./model/netG_model_epoch_${epoch_number}_iter_0.pth"
root_path="/data/users/yanshengyuan/CVPR2020/data/Multi-PIE/FS_aligned"

generation=1 # generation = 1 means using the generator to process the input images

probe_list="/data/users/yanshengyuan/CVPR2020/data/Multi-PIE/FS/multipie_90_test_list.txt"
gallary_list="/data/users/yanshengyuan/CVPR2020/data/Multi-PIE/FS/multipie_gallery_test_list.txt"

python extract_features.py --gpu_ids $gpu_ids --batch_size $batch_size --save_path $save_path --weights $weights \
                           --root_path $root_path --img_list $probe_list --generation $generation

python extract_features.py --gpu_ids $gpu_ids --batch_size $batch_size --save_path $save_path --weights $weights \
                           --root_path $root_path --img_list $gallary_list --generation $generation

cd eval
python evaluation.py --probe_list_path $probe_list --gallery_list_path $gallary_list
