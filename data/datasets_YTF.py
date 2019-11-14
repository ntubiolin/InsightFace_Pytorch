import cv2
import glob
import os.path as op

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, datasets
# from .align import Alignment


class YTFCroppedFacesDataset(Dataset):
    """
        This dataset loads all faces available in YTF, (align them,)
        and transform them into tensors.
        The path for that face is output along with its tensor.
        This is for models to compute all faces' features and store them
        into disks, otherwise the verification testing set contains too many
        repeated faces that should not be computed again and again.

        Note: I do not align the dataset because I downloaded the cropped
        version from https://github.com/huangyangyu/SeqFace#how-to-test,
        which is aligned by the author.
    """
    def __init__(self, data_root, is_ijbb=True):
        self.img_dir = data_root
        self.transforms = transforms.Compose([
            transforms.Resize([112, 112]),
            transforms.ToTensor(),
            transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
        ])
        self.imgs_list = self.loadImgPaths(self.img_dir)
        # self.alignment = Alignment()

    def loadImgPaths(self, img_dir):
        # e.g. dir/Ted_Turner/3/aligned_detect_3.344.jpg
        paths = glob.glob(op.join(img_dir, '*', '*', '*.jpg'))
        return paths

    def __getitem__(self, idx):
        img_path = self.imgs_list[idx]
        try:
            img = Image.open(img_path)
        except Exception as e:
            print(e)
            print(img_path)
        tensor = self.transforms(img)
        return {
            "tensor": tensor,
            "path": img_path,
        }

    def __len__(self):
        return len(self.imgs_list)


class YTFVerificationPathDataset(Dataset):
    """
        This dataset read the match file of verification set in ijb_dataset_root
        (in the `meta` directory, the filename is sth. like
        "ijbc_template_pair_label.txt") and output the cropped faces'
        paths of both enroll_template and verif_template for each match.

        Models outside can use the path information to read their stored
        features and compute the similarity score of enroll_template and
        verif_template.
    """
    def __init__(self, pair_file, img_dir, leave_ratio=1.0):
        self.match = pd.read_csv(pair_file, dtype=str)
        self.img_dir = img_dir
        if leave_ratio < 1.0:  # shrink the number of verified pairs
            indice = np.arange(len(self.match))
            np.random.seed(0)
            np.random.shuffle(indice)
            left_number = int(len(self.match) * leave_ratio)
            self.match = self.match.iloc[indice[:left_number]]

    def __getitem__(self, idx):
        def path_suffixes(id_str):
            paths = glob.glob(op.join(self.img_dir, id_str, '*'))
            paths = [op.join(*path.split('/')[-3:]) for path in paths]
            return paths
        id1 = self.match.iloc[idx][" first name"].strip()
        id2 = self.match.iloc[idx][" second name"].strip()
        return {
            "enroll_template_id": id1,
            "verif_template_id": id2,
            "enroll_path_suffixes": path_suffixes(id1),
            "verif_path_suffixes": path_suffixes(id2),
            # "is_same": self.match.iloc[idx][" is same:"].strip()
            "is_same": self.match.iloc[idx]["corrected labels"].strip()
        }

    def __len__(self):
        return len(self.match)
