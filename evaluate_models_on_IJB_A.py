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
conf = get_config(training=False)
conf.batch_size=20 # Why bs_size can only be the number that divide 6000 well?
learner = face_learner(conf, inference=True)


def l2normalize(x, ord=2, axis=None):
    '''
    x: (# of bs, 512)
    '''
    norm_of_batches = np.linalg.norm(x, ord=ord, axis=axis)
    norm_of_batches = norm_of_batches.reshape(x.shape[0], 1)
    normalized_x = x / norm_of_batches
    return normalized_x


def score_fn_original(feat1, feat2):
    # feat1, feat2.shape (512, x), x is the # of imgs
    return l2normalize(feat1,axis=1).dot(l2normalize(feat2,axis=1).T).mean()


def score_fn_ours(feat_map1s, feat_map2s, learner, attention_strategy='learned', attention_weight=None):
    '''
    feat_map1s & feat_map2s: (# of bs, 32, 7, 7)
    '''
    xCoses = []
    feat_map1s = torch.tensor(feat_map1s).to(conf.device)
    feat_map2s = torch.tensor(feat_map2s).to(conf.device)
    with torch.no_grad():
        for feat_map1 in feat_map1s:
            for feat_map2 in feat_map2s:
                assert len(feat_map1.shape) == 3  # [c, w, h]
                c, w, h = feat_map1.shape
                _feat_map1 = feat_map1.unsqueeze(0)
                _feat_map2 = feat_map2.unsqueeze(0)
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
                    ).type(_feat_map1.type())
                else:
                    raise NotImplementedError
                xCos, attention, cos_patched = learner.get_x_cosine(_feat_map1, _feat_map2, attention)
                xCoses.append(xCos.cpu().numpy())
    xCos_mean = np.array(xCoses).mean()
    return xCos_mean


def evaluate_and_plot(scores, is_sames, nrof_folds=10, dataset_name='AR Face', display_roc_curve=True):
    accuracy, best_threshold, roc_curve_tensor = learner.evaluate_and_plot_roc(scores, is_sames, nrof_folds=10)
    print(f'{dataset_name} - accuray:{accuracy:.5f} (1-{1-accuracy:.5f}), threshold:{best_threshold}')
    if display_roc_curve:
        display(trans.ToPILImage()(roc_curve_tensor))
    return accuracy



'''
For IJB-A, I save features of t1 and t2 "for each comparison" and the corresponding label (is same or not)
'''
def _get_feature(fname):
    npz = np.load(fname)
    return npz['f1'], npz['f2'], npz['same']

def run_IJBA_verification(feat_dir, score_fn, score_fn_kargs, shuffle_order,
                          learner=None, attention_strategy=None, ):
    print(feat_dir)
    is_sames = []
    scores = []
    init_time = time.time()
    fnames = sorted(glob(op.join(feat_dir, '*.npz')))
    # print(len(fnames))
    # fnames = [fnames[i] for i in shuffle_order]
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


def eval3TAR(scores, is_sames):
    for far_target in [1e-1, 1e-2, 1e-3]:
        mean_tar = calculate_val_by_diff(
            thresholds='default',
            dist=(1 - scores),
            actual_issame=is_sames,
            far_target=far_target, ret_acc=True,
            nrof_folds=5
                                        )[0]
        print(f"TAR@FAR{far_target:.4f}:{mean_tar}")


def evaluate_irse50_IJB_A(feat_root_dir):
    # Run evaluation
    model_name = 'ir_se50'
    scores, is_sames = run_IJBA_verification(
        feat_dir=os.path.join(feat_root_dir, f'{model_name}'),
        score_fn=score_fn_original, score_fn_kargs={},
        shuffle_order=shuffle_order
    )

    acc = evaluate_and_plot(scores, is_sames,
                            nrof_folds=10,
                            dataset_name=f'IJB-A {model_name}')

    all_scores['original_ArcFace_irse50'] = scores
    all_acc['original_ArcFace_irse50'] = acc
    eval3TAR(scores, is_sames)


def evaluate_ours_IJB_A(model_name, feat_root_dir):
    fixed_weight = np.load(f'data/correlation_weights/{model_name}.npy')
    fixed_weight /= fixed_weight.sum()

    for attention_strategy in ['uniform', 'learned', 'fixed']:
        print(f'===== attention_strategy: {attention_strategy} =====')
        scores, is_sames = run_IJBA_verification(
            feat_dir=os.path.join(feat_root_dir, f'{model_name}'),
            score_fn=score_fn_ours,
            score_fn_kargs={'learner': learner,
                            'attention_strategy': attention_strategy,
                            'attention_weight': fixed_weight},
            shuffle_order=shuffle_order
        )
        acc = evaluate_and_plot(scores, is_sames,
                                nrof_folds=10,
                                dataset_name=model_name[-7:])

        all_scores[f'{model_name}_{attention_strategy}'] = scores
        all_acc[f'{model_name}_{attention_strategy}'] = acc
        eval3TAR(scores, is_sames)


if __name__ == '__main__':
    # # IJB-A
    # feat_root_dir = 'work_space/IJB_A_features/IJB-A_full_template/split1/'
    cosface_name = \
        '2019-09-02-08-21_accuracy:0.9968333333333333_step:436692_CosFace'
    arcface_name = \
        '2019-08-30-07-36_accuracy:0.9953333333333333_step:655047_None'
    model_names = [cosface_name, arcface_name]

    all_scores = {}
    all_acc = {}

    def record_scores_and_acc(scores, acc, name, name2=None):
        all_scores[name] = scores
        all_acc[name] = acc

    # XXX Why do we need the shuffle order？ Due to the sample ratio issue?
    shuffle_order = np.arange(len(IJBAVerificationDataset(ijba_data_root='data/IJB-A/')))
    np.random.shuffle(shuffle_order)

    def eval_all_my_models(model_names, feat_root_dir):
        for model_name in model_names:
            evaluate_ours_IJB_A(model_name, feat_root_dir)

    # Parse the arguments
    parser = argparse.ArgumentParser(description='evaluation given the feats')
    parser.add_argument('--model', default='all', help='model to test')
    parser.add_argument('--split', default='1', help='split to test')
    parser.add_argument('--feat_root_dir',
                        default='work_space/IJB_A_features/IJB-A_full_template',
                        help='where the features are stored')
    # parser.add_argument('--cmp_strategy', default='mean_comparison',
    #                     help='mean_comparison or compare_only_first_img')
    args = parser.parse_args()
    split_num = int(args.split)
    for i in range(split_num):
        if i == 0:
            continue
        split_name = f'split{i+1}'
        feat_root_dir = op.join(args.feat_root_dir, split_name)
        # comparison_strategy = args.cmp_strategy
        # print('>>>>> Comparison Strategy:', comparison_strategy)
        if(args.model == 'all'):
            evaluate_irse50_IJB_A(feat_root_dir)
            eval_all_my_models(model_names, feat_root_dir)
        elif(args.model == 'irse50'):
            evaluate_irse50_IJB_A(feat_root_dir)
        elif(args.model == 'cosface'):
            evaluate_ours_IJB_A(cosface_name, feat_root_dir)
        elif(args.model == 'arcface'):
            evaluate_ours_IJB_A(arcface_name, feat_root_dir)
        else:
            print('Unknown model name!')

