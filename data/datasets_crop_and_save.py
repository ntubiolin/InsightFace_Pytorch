import os
import cv2
import torch
import os.path as op
import numpy as np
import pandas as pd

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

from .datasets import make_square_box


class CropAndSaveIJBAVeriImgsDataset(Dataset):
    def __init__(self, ijba_data_root='/tmp3/zhe2325138/IJB/IJB-A/',
                 split_name='split1', output_dir='data/IJB_A_cropped_imgs',
                 only_first_image=False, aligned_facial_3points=False):
        self.ijba_data_root = ijba_data_root

        self.output_dir = output_dir
        os.makedirs(op.join(output_dir, 'frames'), exist_ok=True)
        os.makedirs(op.join(output_dir, 'img'), exist_ok=True)
        split_root = op.join(ijba_data_root, 'IJB-A_11_sets', split_name)
        self.only_first_image = only_first_image
        self.metadata = pd.read_csv(op.join(split_root,
                                    f'verify_metadata_{split_name[5:]}.csv'))
        self.metadata = self.metadata.set_index('TEMPLATE_ID')
        self.comparisons = pd.read_csv(op.join(split_root,
                                               f'verify_comparisons_{split_name[5:]}.csv'),
                                       header=None)

        self.transform = transforms.Compose([
            transforms.Resize([112, 112]),
            transforms.ToTensor(),
            transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
        ])

        self.aligned_facial_3points = aligned_facial_3points
        self.src_facial_3_points = self._get_source_facial_3points()

    def _get_source_facial_3points(self, output_size=(112, 112)):
        # set source landmarks based on 96x112 size
        src = np.array([
           [30.2946, 51.6963],  # left eye
           [65.5318, 51.5014],  # right eye
           [48.0252, 71.7366],  # nose
           # [33.5493, 92.3655],  # left mouth
           # [62.7299, 92.2041],  # right mouth
        ], dtype=np.float32)

        # scale landmarkS to match output size
        src[:, 0] *= (output_size[0] / 96)
        src[:, 1] *= (output_size[1] / 112)
        return src

    def _crop_and_save_img_from_entry(self, entry, square=True):
        fname = entry["FILE"]
        if fname[:5] == 'frame':
            fname = 'frames' + fname[5:]  # to fix error in annotation =_=
        output_fpath = op.join(self.output_dir, fname)
        if op.isfile(output_fpath):
            # print('>>> File exists: ', output_fpath)
            return
        img = Image.open(op.join(self.ijba_data_root, 'images', fname)).convert('RGB')

        if self.aligned_facial_3points:
            raise NotImplementedError
        else:
            face_box = [entry['FACE_X'], entry['FACE_Y'], entry['FACE_X'] + entry['FACE_WIDTH'],
                        entry['FACE_Y'] + entry['FACE_HEIGHT']]  # left, upper, right, lower
            face_box = make_square_box(face_box) if square else face_box
            face_img = img.crop(face_box)
            face_img.save(output_fpath)
        return

    def _crop_and_save_imgs_from_entries(self, entries):
        for _, entry in entries.iterrows():
            self._crop_and_save_img_from_entry(entry)

    def __getitem__(self, idx):
        t1, t2 = self.comparisons.iloc[idx]
        t1_entries = self.metadata.loc[[t1]]
        t2_entries = self.metadata.loc[[t2]]
        if self.only_first_image:
            t1_entries, t2_entries = t1_entries.iloc[:1], t2_entries.iloc[:1]

        self._crop_and_save_imgs_from_entries(t1_entries)
        self._crop_and_save_imgs_from_entries(t2_entries)
        # t2_tensors = self._get_tensor_from_entries(t2_entries)
        # if self.only_first_image:
        #     t1_tensors, t2_tensors = t1_tensors.squeeze(0), t2_tensors.squeeze(0)

        # s1, s2 = t1_entries['SUBJECT_ID'].iloc[0], t2_entries['SUBJECT_ID'].iloc[0]
        # is_same = 1 if (s1 == s2) else 0
        # return {
        #     "comparison_idx": idx,
        #     "t1_tensors": t1_tensors,
        #     "t2_tensors": t2_tensors,
        #     "is_same": is_same,
        # }
        return idx

    def __len__(self):
        return len(self.comparisons)
