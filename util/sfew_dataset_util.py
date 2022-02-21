import torch.utils.data as data
import os
import sys
import cv2
import random
import pandas as pd
import numpy as np
import scipy.misc as sm
from torchvision import transforms
import h5py

BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
from util import image_utils 

"""
SFEW dataset
            5         2         1       3      4       0           6    
        Surprise	Fear	Disgust	 Happy	 Sad     Angry	    Neutral	 Sum
Train Set	94(94)	77(78)	50(52)	177(184) 151(161)167(178)  141(144)  857(891)
Val Set	    53(56)  45(46)  23(23)  64(72)   67(73)  76(77)     82(84)   410(431)

RAF dataset
0:Su
1:Fe
2:Di
3:Ha
4:Sa
5:An
6:Ne
"""
to_RAF_label = {
    0:5,
    1:2,
    2:1,
    3:3,
    4:4,
    5:0,
    6:6,
    7:7,
}




class SFEWDataSet(data.Dataset):
    """`SFEW Dataset.
    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, sfew_path, phase='Training', transform=None, basic_aug = False):
        self.transform = transform
        self.phase = phase  # training set or test set
        self.sfew_path = sfew_path
        self.basic_aug = basic_aug
        self.aug_func = [image_utils.flip_image,image_utils.add_gaussian_noise]
        self.data = h5py.File(os.path.join(self.sfew_path, 'SFEW_all_data.h5'), 'r', driver='core')
        # now load the picked numpy arrays
        if self.phase == 'Training':
            self.image_data = self.data['Training_pixel']
            self.label = self.data['Training_label']
            self.image_data = np.asarray(self.image_data)
            self.image_data = self.image_data.reshape((891, 128, 128))
            self.label = np.asarray(self.label)
            for i in range(self.label.shape[0]):
                self.label[i] = to_RAF_label[self.label[i]]
            self.label2 = self.label.copy()

        elif self.phase == 'Val':
            self.image_data = self.data['Val_pixel']
            self.label = self.data['Val_label']
            self.image_data = np.asarray(self.image_data)
            self.image_data = self.image_data.reshape((431, 128, 128))
            self.label = np.asarray(self.label)
            for i in range(self.label.shape[0]):
                self.label[i] = to_RAF_label[self.label[i]]
            self.label2 = self.label.copy()


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        
        img, target, target2 = self.image_data[index], self.label[index], self.label2[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        # img = Image.fromarray(img)
        if self.phase == 'Training':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0,1)
                img = self.aug_func[index](img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target, index, target2

    def __len__(self):
        return len(self.image_data)

if __name__ == '__main__':
    # data_transforms
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02,0.25))])
    sfew_path = '/data/ljj/FER/SFEW2.0'
    dataset = SFEWDataSet(sfew_path, phase = 'Training', transform = data_transforms, basic_aug = True)
    print(dataset.__len__())