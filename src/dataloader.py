import numpy as np
import cv2
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset

class MelanomaDataset(Dataset):
    """
    Custom Dataset class for loading Melanoma Data
    """
    def __init__(self, df, is_train=True, transform=None):
        self.df = df
        self.is_train = is_train
        self.transform = transform
    
    def __getitem__(self, index):
        """
        Get the image path, read it and return it with corresponding label (if is_train=True)
        else return just the image
        """
        self.img_path = self.df.iloc[index]['image_name']
        img = cv2.imread(self.img_path)
        if self.transform:
            img = self.transform(img)
        if self.is_train:
            labl = self.df.iloc[index]['target']
            return img, labl
        else:
            return img
    def __len__(self):
        return len(self.df)