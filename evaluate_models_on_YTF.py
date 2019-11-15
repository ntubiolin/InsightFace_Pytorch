#!/usr/bin/env python
# coding: utf-8
import os
import os.path as op
import argparse
from torchvision import transforms as trans
from tqdm import tqdm
import torch
import numpy as np
import logging

from Learner import face_learner
from config import get_config
from data.datasets_YTF import YTFVerificationPathDataset
from extract_corr_attention_weights import extractAndSaveCorrAttention
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# ## Init Learner and conf
conf = get_config(training=False)
# XXX Why bs_size can only be the number that divide 6000 well?
# conf.batch_size = 20

learner = face_learner(conf, inference=True)


def l2normalize(x, ord=2, axis=None):
    '''
    Input:
        x: of shape (feat dim, )
    '''
    return x / np.linalg.norm(x, ord=ord, axis=axis)


def score_fn_original(feat1, feat2):
    '''
    Input: 
        feat1, feat2: of shape (feat dim, feat dim)
    '''
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
            assert attention_weight.shape[0] == w
            assert attention_weight.shape[1] == h
            attention = torch.tensor(attention_weight).view(
                [1, 1, attention_weight.shape[0], attention_weight.shape[1]]
            ).type(feat_map1.type())
        else:
            raise NotImplementedError
        xCos, attention, cos_patched = \
            learner.get_x_cosine(feat_map1, feat_map2, attention)
    return xCos.cpu().numpy()


def evaluate_and_plot(scores, is_sames, logger, nrof_folds=10,
                      dataset_name='AR Face', display_roc_curve=True,
                      specified_kfolds=None):
    accuracy, best_threshold, roc_curve_tensor = \
            learner.evaluate_and_plot_roc(scores, is_sames,
                                          nrof_folds)
    message = f'{dataset_name} - accuray:{accuracy:.5f} ' \
              f'(1-{1-accuracy:.5f}), threshold:{best_threshold}'
    print(message)
    logger.info(message)
    if display_roc_curve:
        roc_img = trans.ToPILImage()(roc_curve_tensor)
        # display(roc_img)
        # roc_img.save(log_filename[:-4] + '.png')
        roc_img.save(logger.handlers[0].baseFilename[:-4] + '.png')
    return accuracy


def _get_feature_original(feat_dir, suffixes, ends_with='.npy'):
    '''
    Return value:
        feats: it is of shape (# of suffixes, feat dimension)
    '''
    feats = []
    for suffix in suffixes:
        feat = np.load(op.join(feat_dir, suffix[0]) + ends_with)
        feats.append(feat)
    feats = np.array(feats)
    return feats


def _get_feature_ours(feat_dir, suffixes, ends_with='.npz'):
    '''
    Return value:
        feats: it is of shape (# of suffixes, feat dimension)
    '''
    feats = []
    for suffix in suffixes:
        feat = np.load(op.join(feat_dir, suffix[0]) + ends_with)['feat_map']
        feats.append(feat)
    feats = np.array(feats)
    return feats


def run_YTF_verification(loader, feat_dir, compare_strategy, _get_feature,
                         score_fn, score_fn_kargs,
                         logger,
                         learner=None, attention_strategy=None):
    assert compare_strategy in ['compare_only_first_img', 'mean_comparison']
    is_sames = []
    scores = []
    # init_time = time.time()
    ignored_count = 0
    tmp_same_count = 0

    msg_total = f"total number: {len(loader)}"
    print(msg_total)
    # f.write(msg_total + '\n')
    logger.info(msg_total)
    for i, pair in tqdm(enumerate(loader), total=len(loader)):
        # if i % 10000 == 0:
        #     msg = (f"Processing match {i}, "
        #            f"elapsed {time.time() - init_time:.1f} "
        #            f"seconds, positive ratio: {tmp_same_count / (i+1):.4f}")
        #     print(msg)
        #     # f.write(msg + '\n')
        #     logger.info(msg)

        if len(pair['enroll_path_suffixes']) == 0 or\
           len(pair['verif_path_suffixes']) == 0:
            ignored_count += 1
            continue

        if compare_strategy == 'compare_only_first_img':
            enroll_feature = _get_feature(
                feat_dir, pair['enroll_path_suffixes'][:1]).squeeze(0)
            verif_feature = _get_feature(
                feat_dir, pair['verif_path_suffixes'][:1]).squeeze(0)
        elif compare_strategy == 'mean_comparison':
            # print('Warning: using mean_comparison')
            enroll_feature = _get_feature(
                feat_dir, pair['enroll_path_suffixes']).mean(axis=0)
            verif_feature = _get_feature(
                feat_dir, pair['verif_path_suffixes']).mean(axis=0)
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
    msg_ignored = f'{ignored_count} pairs are ignored since '\
                  f'one of the template has no valid image.'
    print(msg_ignored)
    logger.info(msg_ignored)
    ############
    scores = np.array(scores).squeeze()
    is_sames = np.array(is_sames).squeeze().astype(np.bool)
    np.savetxt(f"{logger.handlers[0].baseFilename[:-4]}_scores.csv",
               scores, delimiter=",")
    np.savetxt(f"{logger.handlers[0].baseFilename[:-4]}_is_sames.csv",
               is_sames, delimiter=",")
    return scores, is_sames


# In[7]:


all_scores = {}
all_acc = {}


def record_scores_and_acc(scores, acc, name, name2=None):
    all_scores[name] = scores
    all_acc[name] = acc


# ## Run irse-50 model
def evaluate_irse50_YTF(loader, dataset_name, comparison_strategy, feat_root_dir):
    model_name = f'{dataset_name}_ir_se50'

    log_filename = f"logs/Log_{model_name}.txt"
    logging.basicConfig(filename=log_filename, level=logging.INFO)
    logger = logging.getLogger()

    scores, is_sames = run_YTF_verification(
        # loader, feat_dir=f'work_space/{dataset_name}_features/{model_name}/',
        loader, feat_dir=f'{feat_root_dir}/{model_name}/',
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


def evaluate_ours_YTF(model_name, loader, dataset_name, comparison_strategy,feat_root_dir):
    corr_weight_file = f'data/correlation_weights/{model_name}.npy'
    if not op.isfile(corr_weight_file):
        fixed_weight = extractAndSaveCorrAttention(model_name)
    else:
        fixed_weight = np.load(corr_weight_file)
    fixed_weight /= fixed_weight.sum()

    for attention_strategy in ['uniform', 'learned', 'fixed']:
        print(f'===== attention_strategy: {attention_strategy} =====')
        log_filename = f"logs/Log_{dataset_name}_"\
                       f"{model_name}_{attention_strategy}.txt"
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename=log_filename, level=logging.DEBUG)
        logger = logging.getLogger()
        scores, is_sames = run_YTF_verification(
            loader,
            feat_dir=f'{feat_root_dir}/{dataset_name}_{model_name}/',
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


def eval_all_models(model_names, loader, dataset_name, comparison_strategy, feat_root_dir):
    for model_name in model_names:
        print(f'-------- Evaluating on model {model_name} --------')
        evaluate_ours_YTF(model_name, loader,
                          dataset_name, comparison_strategy, feat_root_dir)


def main():
    # # Evaluate on YTF
    model_names = [
        # AAAI 
        # '2019-09-02-08-21_accuracy:0.9968333333333333_step:436692_CosFace',
        # '2019-08-30-07-36_accuracy:0.9953333333333333_step:655047_None'
        # New, detached models
        # '2019-09-06-08-07_accuracy:0.9970000000000001_step:1601204_CosFace.pth',
        # '2019-09-10-07-18_accuracy:0.9968333333333333_step:946166_ArcFace.pth',
        '2019-11-12-15-54_accuracy:0.99550_step:155260_CosFace_ResNet50_detach_False_MS1M_detachedxCosNoDe',
        '2019-11-12-08-13_accuracy:0.99567_step:154666_ArcFace_ResNet50_detach_False_MS1M_detachedxCosNoDe'
            ]
    # general
    parser = argparse.ArgumentParser(description='evaluation '
                                                 'given the features')
    parser.add_argument('--model', default='all', help='model to test')
    parser.add_argument('--feat_root_dir',
                        default='work_space/features/YTF_features_aligned',
                        help='where the features are stored')
    parser.add_argument('--cmp_strategy', default='mean_comparison',
                        help='mean_comparison or compare_only_first_img')
    args = parser.parse_args()

    feat_root_dir = args.feat_root_dir
    comparison_strategy = args.cmp_strategy
    print('>>>>> Comparison Strategy:', comparison_strategy)
    leave_ratio = 1
    dataset_name = 'YTF'
    loader = torch.utils.data.DataLoader(
        YTFVerificationPathDataset(pair_file=op.join(feat_root_dir,
                                                     'splits_corrected_1.txt'),
                                   img_dir='data/YTF_aligned_SeqFace/',
                                   leave_ratio=leave_ratio),
        batch_size=1, shuffle=False
    )
    if(args.model == 'all'):
        evaluate_irse50_YTF(loader, dataset_name, comparison_strategy, feat_root_dir)
        eval_all_models(model_names, loader, dataset_name, comparison_strategy, feat_root_dir)
    elif(args.model == 'irse50'):
        evaluate_irse50_YTF(loader, dataset_name, comparison_strategy, feat_root_dir)
    elif(args.model == 'cosface'):
        evaluate_ours_YTF(model_names[0], loader, dataset_name,
                          comparison_strategy, feat_root_dir)
    elif(args.model == 'arcface'):
        evaluate_ours_YTF(model_names[1], loader, dataset_name,
                          comparison_strategy, feat_root_dir)
    else:
        print('Unknown model name!')


if __name__ == '__main__':
    main()
