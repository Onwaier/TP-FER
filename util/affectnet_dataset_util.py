import os, sys, shutil
import random as rd
from os import listdir
from PIL import Image
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.utils.data as data
import pdb
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss
import h5py
import torchvision.transforms.functional as tf
from torchvision import transforms
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
from util import image_utils 
from util.cat_AffectNet_csv import single_face_alignment
"""
load AffectNet dataset
标签对应关系
==================================
train dataset
Neutral     ->  0 (74874)
Happy       ->  1 (134415)
Sad         ->  2 (25459)
Surprise    ->  3 (14090)
Fear        ->  4 (6378)
Disgust     ->  5 (3803)
Anger       ->  6 (24882)
Contempt    ->  7 (3750)
None        ->  8 (33088)
Uncertain   ->  9 (11645)
Non-Face    ->  10(82415)

==================================

the struct of datasets is as follow:

- datasets
    - AffectNet 
        - Manually_Annotated_compressed
            - Manually_Annotated_Images
                - 1
                    - 7ffb654b8d3827c453b4a7ffcebd4e4475e33c9097a047d45d38244a.jpg
                    - fd9a175d28d67f44f0d277c23509894e4cad1a17d62dff10094b24e2.JPG
                    ...
                - 2
                - 3
                ...

"""

to_RAF_label = {
    0:6,
    1:3,
    2:4,
    3:0,
    4:1,
    5:2,
    6:5,
    7:7,
}
def read_affectnet_csv(csv_file_path, phase='train', num_classes = 8):
    cnt = 0
    with open(csv_file_path) as f:
        header = ''
        data = []
        for line in f:
            cnt += 1
            if cnt == 1 and phase == 'train':
                header = line
                continue
            str_list = line.strip().split(',')
            if int(str_list[6]) >= num_classes: # 过滤掉非表情和不确定的图片
                continue
            str_list[1: 5] = list(map(int, str_list[1:5]))
            str_list[5] = str_list[5].split(';')
            str_list[5] = list(map(float, str_list[5]))
            str_list[5] = list(map(int, str_list[5]))
            str_list[5] = [(str_list[5][2 * i], str_list[5][2 * i + 1]) for i in range(68)]
            str_list[6] = int(str_list[6])

            data.append(str_list)

        return data

def get_mask(image_path, landmark):
    img = cv2.imread(image_path)
    if img is None:
        print('image_path', image_path)
        return None
    mask = np.zeros(img.shape[:2])
    p = [[[int(e[0]), int(e[1])]] for e in landmark]
    hull = cv2.convexHull(np.array(p), False)
    cv2.fillPoly(mask, [hull], 255)
    return mask



class AffectNetDataset(data.Dataset):
    """`AffectNet Dataset.

    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, affectnet_path, phase='train', transform=None, basic_aug = False, with_align=True, num_classes=8):
        self.transform = transform
        self.phase = phase  # training set or test set
        self.affectnet_path = affectnet_path
        self.basic_aug = basic_aug
        self.with_align = with_align
        self.num_classes = num_classes
        self.aug_func = [image_utils.flip_image,image_utils.add_gaussian_noise]
        BASE_DATASET_PATH = os.path.join(self.affectnet_path, 'Manually_Annotated_file_lists')
        if self.with_align:
            IMAGE_BASE_PATH = os.path.join(self.affectnet_path, 'Manually_Annotated_compressed/Manually_Annotated_Images_Align')
        else:
            IMAGE_BASE_PATH = os.path.join(self.affectnet_path, 'Manually_Annotated_compressed/Manually_Annotated_Images')
        train_dataset_path = os.path.join(BASE_DATASET_PATH, 'training.csv')
        test_dataset_path = os.path.join(BASE_DATASET_PATH, 'validation.csv')
        if self.phase == 'train':
            self.data = read_affectnet_csv(train_dataset_path, phase = 'train', num_classes = num_classes)
        elif self.phase == 'test':
            self.data = read_affectnet_csv(test_dataset_path, phase = 'test', num_classes = num_classes)
        # now load the picked numpy arrays
        origin_image_path = [line[0] for line in self.data]
        self.image_path = [os.path.join(IMAGE_BASE_PATH, origin_image_path[idx]) for idx in range(len(self.data))]
        # self.face_pos = [[self.data[idx][1], self.data[idx][2], self.data[idx][3], self.data[idx][4]] for idx in range(len(self.data))]
        # self.image_landmark = [line[5] for line in self.data]
        # self.label = [to_RAF_label[line[6]] for line in self.data]
        self.label = [line[6] for line in self.data]
        self.label = np.array(self.label)
        self.label2 = self.label.copy()
        self.tmp_label = self.label.copy()
        self.tmp_label2 = self.label2.copy()

    def __getitem__(self, index):
        image = cv2.imread(self.image_path[index])
        image = image[:, :, ::-1]
        label = self.label[index]
        label2 = self.label[index]

        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0,1)
                image = self.aug_func[index](image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label, index, label2

    def __len__(self):
        return len(self.image_path)

    def get_labels(self):
        return self.label

 