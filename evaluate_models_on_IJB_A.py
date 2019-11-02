#!/usr/bin/env python
# coding: utf-8
import os
import os.path as op
import time
import argparse
from glob import glob
from data.data_pipe import get_val_pair
from torchvision import transforms as trans
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

from Learner import face_learner
from config import get_config
from utils import hflip_batch, cosineDim1
from data.datasets import (IJBCAllCroppedFacesDataset, IJBCVerificationPathDataset,
                           ARVerificationAllPathDataset, IJBAVerificationDataset)
from verification import calculate_val_by_diff, calculate_val_far

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# ## Init Learner and conf

# In[2]:


conf = get_config(training=False)
conf.batch_size=20 # Why bs_size can only be the number that divide 6000 well?
learner = face_learner(conf, inference=True)


# In[3]:


def l2normalize(x, ord=2, axis=None):
    return x / np.linalg.norm(x, ord=ord, axis=axis)


def score_fn_original(feat1, feat2):
    # feat1, feat2.shape (512, x), x is the # of imgs
    return l2normalize(feat1).dot(l2normalize(feat2).T).mean()


def score_fn_ours(feat_map1, feat_map2, learner, attention_strategy='learned', attention_weight=None):
    with torch.no_grad():
        assert len(feat_map1.shape) == 3  # [c, w, h]
        c, w, h = feat_map1.shape
        feat_map1 = torch.tensor(feat_map1).unsqueeze(0).to(conf.device)
        feat_map2 = torch.tensor(feat_map2).unsqueeze(0).to(conf.device)
        if attention_strategy == 'learned':
            attention = None
        elif attention_strategy == 'uniform':
            learner.model_attention.eval()
            attention = torch.ones([1, 1, w, h])  # [batch, c, w, h]
            attention /= attention.sum()
            attention = attention.to(conf.device)
        elif attention_strategy == 'fixed':
            assert attention_weight is not None
            assert attention_weight.shape[0] == w and attention_weight.shape[1] == h
            attention = torch.tensor(attention_weight).view(
                [1, 1, attention_weight.shape[0], attention_weight.shape[1]]
            ).type(feat_map1.type())
        else:
            raise NotImplementedError
        xCos, attention, cos_patched = learner.get_x_cosine(feat_map1, feat_map2, attention)
    return xCos.cpu().numpy()

def evaluate_and_plot(scores, is_sames, nrof_folds=10, dataset_name='AR Face', display_roc_curve=True):
    accuracy, best_threshold, roc_curve_tensor = learner.evaluate_and_plot_roc(scores, is_sames, nrof_folds=10)
    print(f'{dataset_name} - accuray:{accuracy:.5f} (1-{1-accuracy:.5f}), threshold:{best_threshold}')
    if display_roc_curve:
        display(trans.ToPILImage()(roc_curve_tensor))
    return accuracy


# # IJB-A
feat_root_dir = 'work_space/IJB_A_features/IJB-A_full_template/split1/'
# In[4]:


all_scores = {}
all_acc = {}

def record_scores_and_acc(scores, acc, name, name2=None):
    all_scores[name] = scores
    all_acc[name] = acc

shuffle_order = np.arange(len(IJBAVerificationDataset()))
np.random.shuffle(shuffle_order)


# In[5]:


'''
For IJB-A, I save features of t1 and t2 "for each comparison" and the corresponding label (is same or not)
'''
def _get_feature(fname):
    npz = np.load(fname)
    return npz['f1'], npz['f2'], npz['same']

def run_IJBA_verification(feat_dir, score_fn, score_fn_kargs, shuffle_order,
                          learner=None, attention_strategy=None, ):
    is_sames = []
    scores = []
    init_time = time.time()
    fnames = sorted(glob(op.join(feat_dir, '*.npz')))
    fnames = [fnames[i] for i in shuffle_order]
    print(len(fnames))
    for i, fname in tqdm(enumerate(fnames), total=len(fnames)):
        # if i % 500 == 0:
        #     print(f"Processing match {i}, elapsed {time.time() - init_time:.1f} seconds")
        f1, f2, is_same = _get_feature(fname)
        score = score_fn(f1, f2, **score_fn_kargs)
        score = score.cpu().numpy() if torch.is_tensor(score) else score
        scores.append(score)
        is_sames.append(is_same.astype(np.bool))

    scores = np.array(scores).squeeze()
    is_sames = np.array(is_sames).squeeze().astype(np.bool)
    return scores, is_sames


# In[6]:


scores, is_sames = run_IJBA_verification(
    feat_dir=os.path.join(feat_root_dir, 'ir_se50/'),
    score_fn=score_fn_original, score_fn_kargs={},
    shuffle_order=shuffle_order
)
print(np.histogram(scores), np.histogram(is_sames))
acc = evaluate_and_plot(scores, is_sames, nrof_folds=10, dataset_name='IJBC')
all_scores['original_ArcFace'] = scores
all_acc['original_ArcFace'] = acc


# In[9]:


for far_target in [1e-1, 1e-2, 1e-3]:
    mean_tar = calculate_val_by_diff(
        thresholds='default', dist=(1 - scores), actual_issame=is_sames, far_target=0.001, ret_acc=True,
        nrof_folds=5
    )[0]
    print("TAR@FAR{far_target:.f}:{mean_tar}")
# calculate_val_far()


# In[ ]:


evaluate_and_plot_roc


# In[11]:


model_name = '2019-09-02-08-21_accuracy:0.9968333333333333_step:436692_CosFace'
fixed_weight = np.load(f'/tmp3/biolin/data/insightFace_pytorch/{model_name}.npy')
fixed_weight /= fixed_weight.sum()

for attention_strategy in ['uniform', 'learned', 'fixed']:
    print(f'===== attention_strategy: {attention_strategy} =====')
    scores, is_sames = run_IJBA_verification(
        feat_dir=os.path.join(feat_root_dir, f'{model_name}'),
        score_fn=score_fn_ours,
        score_fn_kargs={'learner': learner, 'attention_strategy': attention_strategy,
                        'attention_weight': fixed_weight},
        shuffle_order=shuffle_order
    )
    acc = evaluate_and_plot(scores, is_sames, nrof_folds=10, dataset_name='IJBC')

    all_scores[f'{model_name}_{attention_strategy}'] = scores
    all_acc[f'{model_name}_{attention_strategy}'] = acc


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='evaluation given the features')
#     # general
#     parser.add_argument('--model', default='all', help='model to test')
#     parser.add_argument('--feat_root_dir',
#                         default='work_space/YTF_features_aligned',
#                         help='where the features are stored')
#     parser.add_argument('--cmp_strategy', default='mean_comparison',
#                         help='mean_comparison or compare_only_first_img')
#     args = parser.parse_args()
#
#     feat_root_dir = args.feat_root_dir
#     comparison_strategy = args.cmp_strategy
#     print('>>>>> Comparison Strategy:', comparison_strategy)
#     if(args.model == 'all'):
#         eval_all_models()
#     elif(args.model == 'irse50'):
#         evaluate_irse50_YTF()
#     elif(args.model == 'cosface'):
#         evaluate_ours_YTF(model_names[0])
#     elif(args.model == 'arcface'):
#         evaluate_ours_YTF(model_names[1])
#     else:
#         print('Unknown model name!')
