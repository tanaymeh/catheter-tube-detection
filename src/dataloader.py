import numpy as np
import pandas as pd
import os
import cv2

import torch
from torch.utils.data import DataLoader, Dataset

class RANCZRData(Dataset):
    def __init__(self, df, num_classes=5, is_train=True, augments=None, img_size=Config.CFG['img_size'], img_path="../input/ranzcr-clip-catheter-line-classification/train"):
        super().__init__()
        self.df = df.sample(frac=1).reset_index(drop=True)
        self.num_classes = num_classes
        self.is_train = is_train
        self.augments = augments
        self.img_size = img_size
        self.img_path = img_path
    def __getitem__(self, idx):
        image_id = self.df['StudyInstanceUID'].values[idx]
        image = cv2.imread(os.path.join(self.img_path, image_id + ".jpg"))
        image = image[:, :, ::-1]
        
        # Augments must be albumentations
        if self.augments:
            img = self.augments(image=image)['image']
        
        if self.is_train:
            label = self.df[self.df['StudyInstanceUID'] == image_id].values.tolist()[0][1:-1]
            return img, torch.tensor(label)
        
        return img
    
    def __len__(self):
        return len(self.df)