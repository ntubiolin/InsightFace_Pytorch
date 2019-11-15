import os.path as op
import numpy as np
from config import get_config
import argparse
from Learner import face_learner
from data.data_pipe import get_val_pair
from torchvision import transforms as trans
from tqdm import tqdm_notebook as tqdm

from utils import getUnitAttention, getCorrAttention

conf = get_config(training=False)
# XXX Why bs_size can only be the number that divide 6000 well?
conf.batch_size = 200
learner = face_learner(conf, inference=True)


def extractAndSaveCorrAttention(model_name):
    print(f'>>> Extracting corr attention for {model_name}')
    learner.load_state(conf, model_name + '.pth',
                            model_only=True,
                            from_save_folder=True,
                            strict=True, model_atten=True)
    lfw, lfw_issame = get_val_pair(conf.emore_folder, 'lfw')
    print(lfw.shape)
    corrPlot, corr_eff = \
        learner.plot_CorrBtwPatchCosAndGtCos(conf,
                                                lfw, lfw_issame,
                                                nrof_folds=10, tta=True,
                                                attention=None)
    corr_attention = getCorrAttention(corr_eff, conf)
    np.save(op.join('data/correlation_weights', model_name), corr_eff)
    return corr_attention