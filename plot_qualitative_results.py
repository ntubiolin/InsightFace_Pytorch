import os
import argparse
from config import get_config
from Learner import face_learner
from data.data_pipe import get_val_pair


def initialize_learner(conf, mdl_name):
    learner = face_learner(conf, inference=True)
    learner.load_state(conf, mdl_name, model_only=True,
                       from_save_folder=False, strict=True, model_atten=True)
    return learner


def plotResults(conf, learner, exdir, dataset_name='lfw'):
    assert dataset_name in ['lfw', 'agedb_30', 'cfp_fp']
    dataset, dataset_issame = get_val_pair(conf.emore_folder, dataset_name)
    learner.plot_Examples(conf,
                          dataset, dataset_issame,
                          nrof_folds=10, tta=False,
                          attention=None,
                          exDir=exdir)


def main(conf, mdl_name, exdir, dataset_name):
    print(f'>>>> Plot {mdl_name} on {dataset_name}, and save to {exdir}...')
    learner = initialize_learner(conf, mdl_name)
    plotResults(conf, learner, exdir, dataset_name)


if __name__ == "__main__":
    mdl_name_default = '2019-08-25-14-35_accuracy:0.9931666666666666_step:218349_None.pth'
    parser = argparse.ArgumentParser(description='feature extraction')
    # general
    parser.add_argument('--model', default=mdl_name_default,
                        help='model to test')
    parser.add_argument('--dataset', default='lfw',
                        help='plot on which dataset')
    parser.add_argument('--exdir',
                        default='work_space/results/defaultPlotExDir',
                        help='dir to save imgs')
    args = parser.parse_args()

    conf = get_config(training=False)
    # Why bs_size can only be the number that divide 6000 well?
    conf.batch_size = 200

    # exdir = 'cosPatchFtWithMs1M_learnedAtten_LFW_1104_'
    exdir = args.exdir
    dataset_name = args.dataset
    mdl_name = args.model
    os.makedirs(exdir, exist_ok=True)

    main(conf, mdl_name, exdir, dataset_name)
