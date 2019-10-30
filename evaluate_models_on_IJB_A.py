#!/usr/bin/env python
# coding: utf-8
import os
import os.path as op
import time
import argparse
from glob import glob
from data.data_pipe import get_val_pair
from torchvision import transforms as trans
from tqdm import tqdm_notebook as tqdm
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
    return l2normalize(feat1).dot(l2normalize(feat2))


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


# # Evaluate on IJB-C

# In[4]:


loader = torch.utils.data.DataLoader(
    IJBCVerificationPathDataset('/tmp3/zhe2325138/IJB/IJB-C/', occlusion_lower_bound=1, leave_ratio=0.002),
    batch_size=1, shuffle=False
)
len(loader)


# In[5]:


def _get_feature_original(feat_dir, suffixes, ends_with='.npy'):
    return np.concatenate([
        np.load(op.join(feat_dir, suffix[0]) + ends_with)
        for suffix in suffixes
    ], axis=0)


def _get_feature_ours(feat_dir, suffixes, ends_with='.npz'):
#     print(op.join(feat_dir, suffixes[0][0]) + ends_with)
    return np.concatenate([
        np.load(op.join(feat_dir, suffix[0]) + ends_with)['feat_map']
        for suffix in suffixes
    ], axis=0)


def run_IJBC_verification(loader, feat_dir, compare_strategy, _get_feature, score_fn, score_fn_kargs,
                          learner=None, attention_strategy=None):
    assert compare_strategy in ['compare_only_first_img', 'mean_comparison']
    is_sames = []
    scores = []
    init_time = time.time()
    ignored_count = 0
    for i, pair in enumerate(loader):
        if i % 1000 == 0:
            print(f"Processing match {i}, elapsed {time.time() - init_time:.1f} seconds")
        if len(pair['enroll_path_suffixes']) == 0 or len(pair['verif_path_suffixes']) == 0:
            ignored_count += 1
            continue

        if compare_strategy == 'compare_only_first_img':
            enroll_feature = _get_feature(feat_dir, pair['enroll_path_suffixes'][:1]).squeeze(0)
            verif_feature = _get_feature(feat_dir, pair['verif_path_suffixes'][:1]).squeeze(0)
        elif compare_strategy == 'mean_comparison':
            enroll_feature = _get_feature(feat_dir, pair['enroll_path_suffixes']).mean(axis=0)
            verif_feature = _get_feature(feat_dir, pair['verif_path_suffixes']).mean(axis=0)
        else:
            raise NotImplementedError
        score = score_fn(enroll_feature, verif_feature, **score_fn_kargs)
        scores.append(score)
        is_sames.append(pair['is_same'].cpu().numpy().astype(np.bool))

    print(f'{ignored_count} pairs are ignored since one of the template has no valid image.')
    scores = np.array(scores).squeeze()
    is_sames = np.array(is_sames).squeeze().astype(np.bool)
    return scores, is_sames


# In[6]:


all_scores = {}
all_acc = {}

def record_scores_and_acc(scores, acc, name, name2=None):
    all_scores[name] = scores
    all_acc[name] = acc


# In[ ]:


scores, is_sames = run_IJBC_verification(
    loader, feat_dir='./saved_features/IJB-C/ir_se50_original/',
    score_fn=score_fn_original, _get_feature=_get_feature_original,
    compare_strategy='compare_only_first_img', score_fn_kargs={}
)
acc = evaluate_and_plot(scores, is_sames, nrof_folds=10, dataset_name='IJBC')
all_scores['original_ArcFace'] = scores
all_acc['original_ArcFace'] = acc


# In[ ]:


model_name = '2019-09-02-08-21_accuracy:0.9968333333333333_step:436692_CosFace'
fixed_weight = np.load(f'/tmp3/biolin/data/insightFace_pytorch/{model_name}.npy')
fixed_weight /= fixed_weight.sum()

for attention_strategy in ['uniform', 'learned', 'fixed']:
    print(f'===== attention_strategy: {attention_strategy} =====')
    scores, is_sames = run_IJBC_verification(
        loader, feat_dir=f'./saved_features/IJB-C/{model_name}',
        score_fn=score_fn_ours, _get_feature=_get_feature_ours,
        compare_strategy='compare_only_first_img',
        score_fn_kargs={'learner': learner, 'attention_strategy': attention_strategy,
                        'attention_weight': fixed_weight}
    )
    acc = evaluate_and_plot(scores, is_sames, nrof_folds=10, dataset_name='IJBC')

    all_scores[f'{model_name}_{attention_strategy}'] = scores
    all_acc[f'{model_name}_{attention_strategy}'] = acc

# # IJB-A

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
    for i, fname in enumerate(fnames):
        if i % 500 == 0:
            print(f"Processing match {i}, elapsed {time.time() - init_time:.1f} seconds")
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
    feat_dir='./saved_features/IJB-A/split1/ir_se50/',
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
        feat_dir=f'./saved_features/IJB-A/split1/{model_name}',
        score_fn=score_fn_ours,
        score_fn_kargs={'learner': learner, 'attention_strategy': attention_strategy,
                        'attention_weight': fixed_weight},
        shuffle_order=shuffle_order
    )
    acc = evaluate_and_plot(scores, is_sames, nrof_folds=10, dataset_name='IJBC')

    all_scores[f'{model_name}_{attention_strategy}'] = scores
    all_acc[f'{model_name}_{attention_strategy}'] = acc
