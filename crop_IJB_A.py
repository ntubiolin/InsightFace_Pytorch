#!/usr/bin/env python
# coding: utf-8
import os
import time
import os.path as op
import argparse
import torch
import numpy as np
from torchvision import transforms as trans
from tqdm import tqdm

from Learner import face_learner
from config import get_config
from utils import hflip_batch
from data.datasets_crop_and_save import CropAndSaveIJBAVeriImgsDataset
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

# In[4]:


conf = get_config(training=False)
conf.batch_size=20 # Why bs_size can only be the number that divide 6000 well?
learner = face_learner(conf, inference=True)

# # IJB-A


# Extract features for IJB-A dataset
def run_ours_IJBA(loader,
                  img_output_root, split_name,
                  only_first_image=True):
    # if not only_first_image:
    #     assert loader.batch_size == 1

    # dst_dir = op.join(img_output_root,
    #                   f'IJB-A_full_template/{split_name}/{model_name}/')
    # os.makedirs(dst_dir, exist_ok=True)

    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader), total=len(loader)):
            x = 1  # Do nothing!
            # if i % 50 == 0:
            #     print(f"Elapsed {time.time() - begin_time:.1f} seconds.")
            # t1_tensors = batch['t1_tensors']
            # t2_tensors = batch['t2_tensors']

            # if not only_first_image:
            #     t1_tensors, t2_tensors = t1_tensors.squeeze(0), t2_tensors.squeeze(0)

            # comparison_idx = batch['comparison_idx']
            # is_same = batch['is_same']

            # if xCos_or_original == 'xCos':
            #     _, feat1 = process_batch_xcos(t1_tensors, tta=True)
            #     _, feat2 = process_batch_xcos(t2_tensors, tta=True)
            # elif xCos_or_original == 'original':
            #     feat1 = process_batch_original(t1_tensors, tta=True)
            #     feat2 = process_batch_original(t2_tensors, tta=True)
            # else:
            #     raise NotImplementedError

            # if not only_first_image:
            #     feat1, feat2 = feat1.unsqueeze(0), feat2.unsqueeze(0)
#           #   print(feat1.shape, t1_tensors.shape)

            # for idx, f1, f2, same in zip(comparison_idx, feat1, feat2, is_same):
            #     target_path = op.join(dst_dir, str(int(idx.cpu().numpy()))) + '.npz'
            #     if op.exists(target_path):
            #         if i % 50 == 0:
            #             print(f"Skipping {target_path} because it exists.")
            #         continue
            #     # if i % 50 == 0:
            #     #     print(f'Saving to {target_path}')

            #     np.savez(target_path, idx=idx.cpu().numpy(),
            #              same=same.cpu().numpy(),
            #              f1=f1.cpu().numpy(), f2=f2.cpu().numpy(), )

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='feature extraction')
    # general
    parser.add_argument('--split', default='1', help='split to test')
    parser.add_argument('--cropped_img_dir',
                        default='data/IJB_A_cropped_imgs',
                        help='output dir')
    args = parser.parse_args()
    n_splits = int(args.split)

    for i in range(n_splits):
        # Skip split1 because it is done.
        # if i == 0:
        #     continue
        split_num = i + 1
        print(f'Extracting split{split_num}...')
        split_name = f'split{split_num}'
        loader_IJBA = torch.utils.data.DataLoader(
            CropAndSaveIJBAVeriImgsDataset(only_first_image=False,
                                    ijba_data_root='data/IJB-A',
                                    output_dir=args.cropped_img_dir,
                                    split_name=split_name),
            batch_size=1,
            num_workers=1
        )

        run_ours_IJBA(loader_IJBA,
                      args.cropped_img_dir, split_name,
                      only_first_image=False)

