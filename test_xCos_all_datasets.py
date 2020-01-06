import numpy as np
from config import get_config
import argparse
from Learner import face_learner
from data.data_pipe import get_val_pair
from torchvision import transforms as trans
from tqdm import tqdm_notebook as tqdm

from utils import getUnitAttention, getCorrAttention


def testWithDatasets(test_datasets, conf, learner, attention):
    for test_dataset in test_datasets:
        pair_imgs, pair_issame = get_val_pair(conf.emore_folder, test_dataset)
        accuracy, best_threshold, roc_curve_tensor = \
            learner.evaluate_attention(conf,
                                       pair_imgs,
                                       pair_issame,
                                       nrof_folds=10,
                                       tta=True,
                                       attention=attention)
        print(test_dataset +
              ' - accuray:{}, threshold:{}'.format(accuracy, best_threshold))


def testBaselineModel(model_name):
    print(f'>>> Testing model {model_name}')
    learner.load_state(conf, model_name,
                       model_only=True, from_save_folder=True,
                       strict=False, model_atten=False)
    test_datasets = ['lfw', 'vgg2_fp', 'agedb_30',
                     'calfw', 'cfp_ff', 'cfp_fp', 'cplfw']
    testWithDatasets(test_datasets, conf, learner, unit_attention)


def testMyModel(model_name, test_type='atten_xCos'):
    assert test_type in ['patch_xCos', 'corr_xCos', 'atten_xCos']
    print(f'>>> Testing model {model_name} with {test_type}')
    if test_type == 'atten_xCos':
        print('>>> None attention')
        parameters = {
                'attention': None
                }
    elif test_type == 'corr_xCos':
        print('>>> Extracting corr attention')
        learner.load_state(conf, model_name,
                           model_only=True,
                           from_save_folder=True,
                           strict=True, model_atten=True)
        lfw, lfw_issame = get_val_pair(conf.emore_folder, 'lfw')
        corrPlot, corr_eff = \
            learner.plot_CorrBtwPatchCosAndGtCos(conf,
                                                 lfw, lfw_issame,
                                                 nrof_folds=10, tta=True,
                                                 attention=None)
        corr_attention = getCorrAttention(corr_eff, conf)
        print('>>> Corr attention extracted')
        parameters = {
                'attention': corr_attention
                }
    elif test_type == 'patch_xCos':
        parameters = {
                'attention': unit_attention
                }
    learner.load_state(conf, model_name,
                       model_only=True, from_save_folder=True,
                       strict=True, model_atten=True)

    test_datasets = ['lfw', 'vgg2_fp', 'agedb_30',
                     'calfw', 'cfp_ff', 'cfp_fp', 'cplfw']
    testWithDatasets(test_datasets, conf, learner, parameters['attention'])


conf = get_config(training=False)
# XXX Why bs_size can only be the number that divide 6000 well?
conf.batch_size = 200
# conf.net_depth = 100
unit_attention = getUnitAttention(conf)
learner = face_learner(conf, inference=True)

# model_name = '2019-11-12-03-59_accuracy:0.9269999999999999_step:191058_CosFace_ResNet50_detach_False_MS1M_detachedtwcc.pth'
# model_name = 'ir_se50.pth'
# model_name = '2019-11-12-04-06_accuracy:0.9301428571428572_step:172029_CosFace_ResNet100_detach_False_MS1M_detachedtwcc.pth'
# model_name = '2019-11-11-17-07_accuracy:0.9135714285714286_step:132330_CosFace_ResNet100_detach_False_MS1M_detachedtwcc.pth'
# model_name = '2019-11-12-16-09_accuracy:0.8971_step:191058_ArcFace_ResNet50_detach_True_MS1M_detachedreproduce.pth'
# testBaselineModel(model_name)

# model_name = '2019-11-11-18-58_accuracy:0.99533_step:100078_ArcFace_ResNet50_detach_False_MS1M_detachedxCosNoDe.pth'
model_name = '2019-09-02-08-21_accuracy:0.9968333333333333_step:436692_CosFace.pth'
# model_name = '2019-09-06-08-07_accuracy:0.9970000000000001_step:1601204_CosFace.pth'
# model_name = '2019-11-15-12-49_accuracy:0.99650_step:327519_CosFace_ResNet50_detach_False_MS1M_detachedxCosNoDeL1.pth'
# model_name = '2019-11-15-15-25_accuracy:0.99583_step:363910_ArcFace_ResNet50_detach_False_MS1M_detachedxCosNoDeL1.pth'
# model_name = '2019-11-15-18-02_accuracy:0.99500_step:400301_ArcFace_ResNet50_detach_False_MS1M_detachedxCosNoDeL1.pth'
# model_name = '2019-11-12-08-13_accuracy:0.99567_step:154666_ArcFace_ResNet50_detach_False_MS1M_detachedxCosNoDe.pth'
model_name = '2019-11-12-16-32_accuracy:0.99183_step:168207_CosFace_ResNet100_detach_False_MS1M_detachedxCosNoDe.pth'
model_name = '2019-08-25-14-35_accuracy:0.9931666666666666_step:218349_None.pth'
model_name = '2019-08-30-07-36_accuracy:0.9953333333333333_step:655047_None.pth'
model_name = '2019-11-12-17-02_accuracy:0.99500_step:191058_ArcFace_ResNet50_detach_False_MS1M_detachedxCosNoDe.pth'
test_types = ['atten_xCos', 'corr_xCos', 'patch_xCos']
for test_type in test_types:
    testMyModel(model_name, test_type)
