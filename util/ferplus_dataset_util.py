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
FERPlus label:emotion
emotions = {
    '0':'angry', #生气
    '1':'disgust', #厌恶
    '2':'fear', #恐惧
    '3':'happy', #开心
    '4':'sad', #伤心
    '5':'surprise', #惊讶
    '6':'neutral', #中性
}
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

def process_data(emotion_raw):
    """
    refer to https://github.com/microsoft/FERPlus/blob/master/src/ferplus.py
    """  
    size = len(emotion_raw)
    emotion_unknown     = [0.0] * size
    emotion_unknown[-2] = 1.0

    # remove emotions with a single vote (outlier removal) 
    for i in range(size):
        if emotion_raw[i] < 1.0 + sys.float_info.epsilon:
            emotion_raw[i] = 0.0

    sum_list = sum(emotion_raw)
    emotion = [0.0] * size
    maxval = max(emotion_raw) 
    if maxval >= 0.5 * sum_list:
        return np.argmax(emotion_raw), np.argmax(emotion_raw) 
    else: 
        emotion_arr = np.array(emotion_raw)
        top_k_idx = emotion_arr.argsort()[::-1][0:2]
        return top_k_idx[0], top_k_idx[1]
        



class FERplusDataSet(data.Dataset):
    def __init__(self, ferplus_path, phase, transform = None, basic_aug = False):
        self.phase = phase
        self.transform = transform
        self.ferplus_path = ferplus_path
        self.file_paths = []
        self.label = []
        self.label2 = []

        # the header of fer2013_new.csv: Usage,Image name,neutral,happiness,surprise,sadness,anger,disgust,fear,contempt,unknown,NF
        label_info = pd.read_csv(os.path.join(self.ferplus_path, 'fer2013new.csv'))
        faces_data = pd.read_csv(os.path.join(self.ferplus_path, 'fer2013.csv'))
        for index in range(len(label_info)):
            usage = label_info.loc[index][0]
            image_name = label_info.loc[index][1]
            if type(image_name) == float or image_name.strip() == '':
                continue
            image_path = os.path.join(self.ferplus_path, usage, image_name)
            # print('image_path', image_path)
            if usage == self.phase:
                # label1, label2 = process_data(label_info.loc[index].tolist()[2:])
                # if label1 >= 8: # unknown or not face
                #     continue
                self.file_paths.append(image_path)
                old_label = int(faces_data.loc[index][0])
                self.label.append(to_RAF_label[old_label])
                self.label2.append(to_RAF_label[old_label])

        self.label = np.array(self.label)
        self.label2 = np.array(self.label2)
        self.basic_aug = basic_aug
        self.aug_func = [image_utils.flip_image,image_utils.add_gaussian_noise]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        # image = image[:, :, np.newaxis]
        # image = np.concatenate((image, image, image), axis = 2)
        image = image[:, :, ::-1] # BGR to RGB
        label, label2 = self.label[idx], self.label2[idx]
        # augmentation
        if self.phase == 'Training':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0,1)
                image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label, idx, label2

class FER2013DataSet(data.Dataset):
    """`FER2013 Dataset.
    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, fer2013_path, phase='Training', transform=None):
        self.transform = transform
        self.phase = phase  # training set or test set
        self.fer2013_path = fer2013_path
        self.data = h5py.File(os.path.join(self.fer2013_path, 'fer2013.h5'), 'r', driver='core')
        # now load the picked numpy arrays
        if self.phase == 'Training':
            self.image_data = self.data['Training_pixel']
            self.label = self.data['Training_label']
            self.image_data = np.asarray(self.image_data)
            self.image_data = self.image_data.reshape((28709, 48, 48))
            self.label = np.asarray(self.label)
            self.label2 = self.label.copy()

        elif self.phase == 'PublicTest':
            self.image_data = self.data['PublicTest_pixel']
            self.label = self.data['PublicTest_label']
            self.image_data = np.asarray(self.image_data)
            self.image_data = self.image_data.reshape((3589, 48, 48))
            self.label = np.asarray(self.label)
            self.label2 = self.label.copy()

        else:
            self.image_data = self.data['PrivateTest_pixel']
            self.label = self.data['PrivateTest_label']
            self.image_data =  np.asarray(self.image_data)
            self.image_data = self.image_data.reshape((3589, 48, 48))
            self.label = np.asarray(self.label)
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
    ferplus_path = '/data/ljj/FER'
    dataset = FERplusDataSet(ferplus_path, phase = 'Training', transform = data_transforms, basic_aug = True)
    print(dataset.__len__())