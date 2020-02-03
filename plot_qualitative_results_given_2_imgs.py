import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms

from config import get_config
from Learner import face_learner
from data.data_pipe import get_val_pair

from mtcnn_pytorch.crop_and_aligned import mctnn_crop_face


def initialize_learner(conf, mdl_name):
    learner = face_learner(conf, inference=True)
    learner.load_state(conf, mdl_name, model_only=True,
                       from_save_folder=False, strict=True, model_atten=True)
    return learner


def plotResults(conf, learner, exdir, img1, img2,
                filename, dataset_name='lfw'):
    transforms_mine = transforms.Compose([
        transforms.Resize([112, 112]),
        transforms.ToTensor(),
        transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
    ])

    assert dataset_name in ['lfw', 'agedb_30', 'cfp_fp']
    # dataset, dataset_issame = get_val_pair(conf.emore_folder, dataset_name)
    img1 = Image.open(img1).convert('RGB')
    img2 = Image.open(img2).convert('RGB')
    # In fact, BGR2RGB=True turn RGB into BGR
    img1 = mctnn_crop_face(img1, BGR2RGB=True)
    img2 = mctnn_crop_face(img2, BGR2RGB=True)
    # img1 = cv2.imread(img1)
    # img2 = cv2.imread(img2)
    # img1 = Image.fromarray(img1)
    # img2 = Image.fromarray(img2)
    img1 = transforms_mine(img1)
    img2 = transforms_mine(img2)

    # img1 = np.array(img1)
    # img2 = np.array(img2)

    # img1 = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)
    # img2 = cv2.resize(img2, dim, interpolation=cv2.INTER_AREA)

    # dataset = np.array([img1, img2])
    # XXX It causes the meta and image returned with one reduntdant result
    dataset = torch.stack([img1, img2, img1, img2])
    print(dataset.size())
    dataset_issame = np.array([1, 1])
    img_base64, meta = learner.plot_Examples(conf,
                                       dataset, dataset_issame,
                                       nrof_folds=10, tta=False,
                                       attention=None,
                                       exDir=exdir,
                                       filename=filename)
    
    return img_base64, meta


def getCroppedTensorFromFilename(filename, transform):
    img = Image.open(filename).convert('RGB')
    img = mctnn_crop_face(img)
    img = transform(img)
    return img


def getPairedTensors(query_filename, filesToCompare):
    transforms_mine = transforms.Compose([
        transforms.Resize([112, 112]),
        transforms.ToTensor(),
        transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
    ])
    image_stack = []
    query_img_tensor = getCroppedTensorFromFilename(query_filename,
                                                    transforms_mine)
    for filename in filesToCompare:
        target_img_tensor = getCroppedTensorFromFilename(filename,
                                                         transforms_mine)
        image_stack.append(query_img_tensor)
        image_stack.append(target_img_tensor)
    image_stack = torch.stack(image_stack)
    return image_stack


def main(conf, mdl_name, exdir, dataset_name, img1, img2, filename):
    print(f'>>>> Plot comparison of {img1} and {img2} '
          '{mdl_name} on {dataset_name}, and save to {exdir}...')
    learner = initialize_learner(conf, mdl_name)
    plotResults(conf, learner, exdir, img1, img2, filename, dataset_name)


if __name__ == "__main__":
    mdl_name_default = '2019-08-25-14-35_accuracy:0.9931666666666666_step:218349_None.pth'
    parser = argparse.ArgumentParser(description='feature extraction')
    # general
    parser.add_argument('--model', default=mdl_name_default,
                        help='model to test')
    parser.add_argument('--dataset', default='lfw',
                        help='plot on which dataset')
    parser.add_argument('--img1', default='gakki1.jpg',
                        help='img to test')
    parser.add_argument('--img2', default='gakki2.jpg',
                        help='img to test')
    parser.add_argument('--filename', default='result.jpg',
                        help='img to test')
    parser.add_argument('--exdir',
                        default='work_space/results/defaultPlotExDir_1124',
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

    main(conf, mdl_name, exdir, dataset_name, args.img1, args.img2, args.filename)
