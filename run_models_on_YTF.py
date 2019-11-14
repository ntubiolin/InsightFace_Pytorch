#!/usr/bin/env python
# coding: utf-8
import os
import time
import os.path as op
import argparse
from torchvision import transforms as trans
from tqdm import tqdm
import torch
import numpy as np
from IPython.display import display

from Learner import face_learner
from config import get_config
from utils import hflip_batch, cosineDim1
from data.datasets_YTF import YTFCroppedFacesDataset#, IJBCVerificationPathDataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def process_batch_original(batch, tta=False):
    if tta:
        fliped = hflip_batch(batch)
        feat_orig = learner.model.get_original_feature(batch.to(conf.device))
        feat_flip = learner.model.get_original_feature(fliped.to(conf.device))
        feat = (feat_orig + feat_flip) / 2
    else:
        feat = learner.model.get_original_feature(batch.to(conf.device))
    return feat


def process_batch_xcos(batch, tta=False):
    if tta:
        fliped = hflip_batch(batch)
        flattened_feature1, feat_map1 = learner.model(batch.to(conf.device))
        flattened_feature2, feat_map2 = learner.model(fliped.to(conf.device))
        feat_map = (feat_map1 + feat_map2) / 2
        flattened_feature = (flattened_feature1 + flattened_feature2) / 2
    else:
        flattened_feature, feat_map = learner.model(batch.to(conf.device))
    return flattened_feature, feat_map


# ## Init Learner and conf
conf = get_config(training=False)
conf.batch_size=20 # Why bs_size can only be the number that divide 6000 well?
learner = face_learner(conf, inference=True)
# # Process YTF data and save them
dataset_name = 'YTF'
loader = torch.utils.data.DataLoader(
    # YTFCroppedFacesDataset('data/YTF_aligned_SeqFace'),
    YTFCroppedFacesDataset('/home/r07944011/datasets_ssd/face/YTF/align_InsightFace/cropped_and_aligned'),
    batch_size=320
)

# ## Original model
def run_irse50_YTF():
    learner.load_state(conf, 'ir_se50.pth', model_only=True,
                       from_save_folder=True, strict=False,
                       model_atten=False)
    learner.model.eval()
    learner.model_attention.eval()

    dst_dir = op.join(feat_root_dir, f'{dataset_name}_ir_se50')
    os.makedirs(dst_dir, exist_ok=True)

    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader), total=len(loader)):
            tensor = batch['tensor']
            batch_size = tensor.size(0)
            src_path = batch['path']
            # if i % 200 == 0:
            #     print(f'Processing {src_path[0]}')
            feat = process_batch_original(tensor, tta=True)
            feat = feat.cpu().numpy()
            for j in range(batch_size):
                os.makedirs(op.join(dst_dir,
                                    *src_path[j].split('/')[-3:-1]),
                            exist_ok=True)
                target_path = op.join(dst_dir, *src_path[j].split('/')[-3:]) + '.npy'
                if op.exists(target_path):
                    if i % 200 == 0:
                        print(f"Skipping {target_path} because it exists.")
                    continue
                np.save(target_path, feat[j])


# ## Our model
def run_ours_YTF(model_name):
    learner.load_state(
        conf, f'{model_name}.pth',
        model_only=True, from_save_folder=True, strict=True, model_atten=True)
    learner.model.eval()
    learner.model.returnGrid = True  # Remember to reset this before return!
    learner.model_attention.eval()

    dst_dir = op.join(feat_root_dir, f'{dataset_name}_{model_name}')
    os.makedirs(dst_dir, exist_ok=True)

    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader), total=len(loader)):
            tensor = batch['tensor']
            src_path = batch['path']
            os.makedirs(op.join(dst_dir, *src_path[0].split('/')[-3:-1]), exist_ok=True)
            target_path = op.join(dst_dir, *src_path[0].split('/')[-3:]) + '.npz'
            # if i % 2000 == 0:
            #    print(f'Processing {src_path[0]} \nSaving to {target_path}')
            if op.exists(target_path):
                if i % 2000 == 0:
                    print(f"Skipping {target_path} because it exists.")
                continue
            flattened_feature, feat_map = process_batch_xcos(tensor, tta=True)
            np.savez(target_path, flattened_feature=flattened_feature.cpu().numpy(),
                     feat_map=feat_map.cpu().numpy())


model_names = [
    # '2019-09-01-15-30_accuracy:0.9946666666666667_step:218346_CosFace',
    '2019-09-02-08-21_accuracy:0.9968333333333333_step:436692_CosFace',
    # '2019-08-25-14-35_accuracy:0.9931666666666666_step:218349_None',
    '2019-08-30-07-36_accuracy:0.9953333333333333_step:655047_None'
]


def run_all_my_models():
    for model_name in model_names:
        print(f'-------- Runing on model {model_name} --------')
        run_ours_YTF(model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='feature extraction')
    # general
    parser.add_argument('--model', default='all', help='model to test')
    parser.add_argument('--feat_root_dir',
                        default='work_space/features/YTF_features_aligned',
                        help='model to test')
    args = parser.parse_args()
    feat_root_dir = args.feat_root_dir
    os.makedirs(feat_root_dir, exist_ok=True)

    if(args.model == 'all'):
        run_all_my_models()
        run_irse50_YTF()
    elif(args.model == 'irse50'):
        run_irse50_YTF()
    elif(args.model == 'cosface'):
        run_ours_YTF(model_names[0])
    elif(args.model == 'arcface'):
        run_ours_YTF(model_names[1])
    else:
        print('Unknown model name!')
