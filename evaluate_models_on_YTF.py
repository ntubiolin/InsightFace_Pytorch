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
import logging
from IPython.display import display

from Learner import face_learner
from config import get_config
from utils import hflip_batch, cosineDim1
from data.datasets_YTF import YTFVerificationPathDataset
from verification import calculate_val_by_diff, calculate_val_far

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# ## Init Learner and conf
conf = get_config(training=False)
conf.batch_size=20 # Why bs_size can only be the number that divide 6000 well?
learner = face_learner(conf, inference=True)


def l2normalize(x, ord=2, axis=None):
    return x / np.linalg.norm(x, ord=ord, axis=axis)


def score_fn_original(feat1, feat2):
    return l2normalize(feat1).dot(l2normalize(feat2))


def score_fn_ours(feat_map1, feat_map2, learner,
                  attention_strategy='learned',
                  attention_weight=None):
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

def evaluate_and_plot(scores, is_sames, logger, nrof_folds=10,
                      dataset_name='AR Face', display_roc_curve=True):
    accuracy, best_threshold, roc_curve_tensor = learner.evaluate_and_plot_roc(scores, is_sames, nrof_folds=10)
    message = f'{dataset_name} - accuray:{accuracy:.5f} (1-{1-accuracy:.5f}), threshold:{best_threshold}'
    print(message)
    logger.info(message)
    #with open(log_filename, 'a') as f:
    #    f.write(message + '\n')
    if display_roc_curve:
        roc_img = trans.ToPILImage()(roc_curve_tensor)
        # display(roc_img)
        # roc_img.save(log_filename[:-4] + '.png')
        roc_img.save(logger.handlers[0].baseFilename[:-4] + '.png')
    return accuracy


# # Evaluate on YTF
leave_ratio = 1
dataset_name = 'YTF'
loader = torch.utils.data.DataLoader(
    YTFVerificationPathDataset(pair_file='data/YTF_aligned_SeqFace/splits.txt',
                               img_dir='data/YTF_aligned_SeqFace/',
                               leave_ratio=leave_ratio),
    batch_size=1, shuffle=False
)

# In[6]:


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


def run_YTF_verification(loader, feat_dir, compare_strategy, _get_feature,
                          score_fn, score_fn_kargs,
                          logger,
                          learner=None, attention_strategy=None):
    assert compare_strategy in ['compare_only_first_img', 'mean_comparison']
    is_sames = []
    scores = []
    init_time = time.time()
    ignored_count = 0
    tmp_same_count = 0

    msg_total = f"total number: {len(loader)}"
    print(msg_total)
    # f.write(msg_total + '\n')
    logger.info(msg_total)
    for i, pair in tqdm(enumerate(loader), total=len(loader)):
        # if i % 10000 == 0:
        #     msg = (f"Processing match {i}, elapsed {time.time() - init_time:.1f}            seconds, positive ratio: {tmp_same_count / (i+1):.4f}")
        #     print(msg)
        #     # f.write(msg + '\n')
        #     logger.info(msg)

        if len(pair['enroll_path_suffixes']) == 0 or len(pair['verif_path_suffixes']) == 0:
            ignored_count += 1
            continue

        if compare_strategy == 'compare_only_first_img':
            enroll_feature = _get_feature(feat_dir, pair['enroll_path_suffixes'][:1]).squeeze(0)
            verif_feature = _get_feature(feat_dir, pair['verif_path_suffixes'][:1]).squeeze(0)
        elif compare_strategy == 'mean_comparison':
            # print('Warning: using mean_comparison')
            enroll_feature = _get_feature(feat_dir, pair['enroll_path_suffixes']).mean(axis=0)
            verif_feature = _get_feature(feat_dir, pair['verif_path_suffixes']).mean(axis=0)
        else:
            raise NotImplementedError
        score = score_fn(enroll_feature, verif_feature, **score_fn_kargs)
        scores.append(score)
        # XXX: Why is pair['is_same'] a list?
        # if bool(int(pair['is_same'][0])):
        #     print((int(pair['is_same'][0])))
        #     print('    >>>', bool(int(pair['is_same'][0])))
        #     print(pair)
        is_same_label = bool(int(pair['is_same'][0]))
        is_sames.append(is_same_label)
        if is_same_label:
            tmp_same_count += 1
    msg_ignored = f'{ignored_count} pairs are ignored since one of the template has no valid image.'
    print(msg_ignored)
    #f.write(msg_ignored + '\n')
    logger.info(msg_ignored)
    ############
    scores = np.array(scores).squeeze()
    is_sames = np.array(is_sames).squeeze().astype(np.bool)
    np.savetxt(f"{logger.handlers[0].baseFilename[:-4]}_scores.csv", scores, delimiter=",")
    np.savetxt(f"{logger.handlers[0].baseFilename[:-4]}_is_sames.csv", is_sames, delimiter=",")
    return scores, is_sames


# In[7]:


all_scores = {}
all_acc = {}


def record_scores_and_acc(scores, acc, name, name2=None):
    all_scores[name] = scores
    all_acc[name] = acc


# ## Run irse-50 model
def evaluate_irse50_YTF():
    model_name = f'{dataset_name}_ir_se50'

    log_filename = f"logs/Log_{model_name}.txt"
    logging.basicConfig(filename=log_filename, level=logging.INFO)
    logger = logging.getLogger()

    scores, is_sames = run_YTF_verification(
        loader, feat_dir=f'work_space/{dataset_name}_features/{model_name}/',
        score_fn=score_fn_original, _get_feature=_get_feature_original,
        compare_strategy=comparison_strategy, score_fn_kargs={},
        logger=logger,
    )
    acc = evaluate_and_plot(scores, is_sames,
                            nrof_folds=10,
                            dataset_name=model_name,
                            logger=logger)
    all_scores['original_ArcFace'] = scores
    all_acc['original_ArcFace'] = acc

# ## Run CosFace model

# In[ ]:

model_names = [
    '2019-09-02-08-21_accuracy:0.9968333333333333_step:436692_CosFace',
    '2019-08-30-07-36_accuracy:0.9953333333333333_step:655047_None'
        ]


def evaluate_ours_YTF(model_name):
    fixed_weight = np.load(f'data/correlation_weights/{model_name}.npy')
    fixed_weight /= fixed_weight.sum()

    for attention_strategy in ['uniform', 'learned', 'fixed']:
        print(f'===== attention_strategy: {attention_strategy} =====')
        log_filename = f"logs/Log_{dataset_name}_{model_name}_{attention_strategy}.txt"
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename=log_filename, level=logging.DEBUG)
        logger = logging.getLogger()
        scores, is_sames = run_YTF_verification(
            loader,
            feat_dir=f'work_space/{dataset_name}_features/{dataset_name}_{model_name}/',
            score_fn=score_fn_ours, _get_feature=_get_feature_ours,
            compare_strategy=comparison_strategy,
            score_fn_kargs={'learner': learner,
                            'attention_strategy': attention_strategy,
                            'attention_weight': fixed_weight},
            logger=logger
        )
        acc = evaluate_and_plot(scores, is_sames, nrof_folds=10,
                                dataset_name=f"{dataset_name}_{model_name}",
                                logger=logger)

        all_scores[f'{model_name}_{attention_strategy}'] = scores
        all_acc[f'{model_name}_{attention_strategy}'] = acc
        print()

# In[ ]:
#
#
# all_scores
#
#
# # In[ ]:
#
#
# all_acc


# ## Run ArcFace model

# In[ ]:


# model_name = '2019-08-30-07-36_accuracy:0.9953333333333333_step:655047_None'
# fixed_weight = np.load(f'data/correlation_weights/{model_name}.npy')
# fixed_weight /= fixed_weight.sum()
#
# for attention_strategy in ['uniform', 'learned', 'fixed']:
#     print(f'===== attention_strategy: {attention_strategy} =====')
#     log_filename = f"logs/Log_{IJB_B_or_C}_{model_name}_{attention_strategy}.txt"
#     for handler in logging.root.handlers[:]:
#         logging.root.removeHandler(handler)
#     logging.basicConfig(filename=log_filename, level=logging.DEBUG)
#     logger = logging.getLogger()
#     scores, is_sames = run_YTF_verification(
#         loader, feat_dir=f"work_space/IJB_features/{IJB_B_or_C}_{model_name}/loose_crop",
#         score_fn=score_fn_ours, _get_feature=_get_feature_ours,
#         compare_strategy='compare_only_first_img',
#         score_fn_kargs={'learner': learner, 'attention_strategy': attention_strategy,
#                         'attention_weight': fixed_weight},
#         logger = logger
#     )
#     acc = evaluate_and_plot(scores, is_sames, nrof_folds=10, dataset_name=f"{IJB_B_or_C}_{model_name}",
#                             logger=logger)
#
#     all_scores[f'{model_name}_{attention_strategy}'] = scores
#     all_acc[f'{model_name}_{attention_strategy}'] = acc


def eval_all_models():
    for model_name in model_names:
        print(f'-------- Evaluating on model {model_name} --------')
        evaluate_ours_YTF(model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluation given the features')
    # general
    parser.add_argument('--model', default='all', help='model to test')
    parser.add_argument('--feat_root_dir',
                        default='work_space/YTF_features_aligned',
                        help='where the features are stored')
    parser.add_argument('--cmp_strategy', default='mean_comparison',
                        help='mean_comparison or compare_only_first_img')
    args = parser.parse_args()

    feat_root_dir = args.feat_root_dir
    comparison_strategy = args.cmp_strategy
    print('>>>>> Comparison Strategy:', comparison_strategy)
    if(args.model == 'all'):
        eval_all_models()
    elif(args.model == 'irse50'):
        evaluate_irse50_YTF()
    elif(args.model == 'cosface'):
        evaluate_ours_YTF(model_names[0])
    elif(args.model == 'arcface'):
        evaluate_ours_YTF(model_names[1])
    else:
        print('Unknown model name!')
