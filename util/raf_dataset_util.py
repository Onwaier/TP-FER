import torch.utils.data as data
import os
import sys
import cv2
import random
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
from util import image_utils 


class RAFDataSet(data.Dataset):
    def __init__(self, raf_path, phase, transform = None, basic_aug = False, noise_file = ''):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path
        self.noise_file = noise_file

        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        if self.noise_file == '':
            df = pd.read_csv(os.path.join(self.raf_path, 'list_patition_label.txt'), sep=' ', header=None)
        else:
            df = pd.read_csv(os.path.join(self.raf_path, self.noise_file), sep=' ', header=None)
        if phase == 'train':
            dataset = df[df[NAME_COLUMN].str.startswith('train')]
        else:
            dataset = df[df[NAME_COLUMN].str.startswith('test')]
        file_names = dataset.iloc[:, NAME_COLUMN].values
        self.label = dataset.iloc[:, LABEL_COLUMN].values - 1 # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
        self.label2 = self.label.copy() 
        self.file_paths = []
        # use raf aligned images for training/testing
        for f in file_names:
            f = f.split(".")[0]
            f = f + "_aligned.jpg"
            path = os.path.join(self.raf_path, 'aligned', f)
            self.file_paths.append(path)
        
        self.basic_aug = basic_aug
        self.aug_func = [image_utils.flip_image,image_utils.add_gaussian_noise]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1] # BGR to RGB
        label = self.label[idx]
        label2 = self.label2[idx]
        # augmentation
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0,1)
                image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label, idx, label2
