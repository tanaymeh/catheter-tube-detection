import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import cv2
from tqdm.notebook import tqdm
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.nn.modules.loss import _WeightedLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from src.augmentations import Augments
from src.dataloader import RANCZRData
from src.models import NFNetModel
from src.trainer import Trainer

warnings.simplefilter("ignore")

class Config:
    CFG = {
        'img_size': 224,
    }

# Change these as you see fit
nb_epochs = 5
device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

data = pd.read_csv("../input/ranzcr-clip-catheter-line-classification/train.csv")
data = data.sample(frac=1).reset_index(drop=True)

# 27,583 in Train, 2500 in Valid
train_split = data[2500:]
valid_split = data[:2500]

print(f"{'='*40}")
print(f"Training on: {train_split.shape[0]} samples")
print(f"Validating on: {valid_split.shape[0]} samples")
print(f"{'='*40}")

if __name__ == "__main__":
    train_set = RANCZRData(df=train_split, augments=Augments.train_augments)
    valid_set = RANCZRData(df=valid_split, augments=Augments.valid_augments)

    train = DataLoader(
        train_set,
        batch_size=16,
        shuffle=True,
        pin_memory=False,
        drop_last=False,
        num_workers=8
    )

    valid = DataLoader(
        valid_set,
        batch_size=32,
        shuffle=False,
        pin_memory=False,
        num_workers=8
    )

    model = NFNetModel().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.001)
    loss_fn_train = nn.BCEWithLogitsLoss()
    loss_fn_val = nn.BCEWithLogitsLoss()

    trainer = Trainer(
        train_dataloader=train,
        valid_dataloader=valid,
        model=model,
        optimizer=optim,
        loss_fn=loss_fn_train,
        val_loss_fn=loss_fn_val,
        scheduler=None,
        device=device,
    )

    train_losses = []
    valid_losses = []

    scaler = GradScaler()

    for epoch in range(nb_epochs):
        print(f"{'-'*20} EPOCH: {epoch+1}/{nb_epochs} {'-'*20}")

        # Run one training epoch
        current_train_loss = trainer.train_one_cycle()
        train_losses.append(current_train_loss)

        # Run one validation epoch
        current_val_loss, op_model = trainer.valid_one_cycle()
        valid_losses.append(current_val_loss)

        # Empty CUDA cache
        torch.cuda.empty_cache()
        
        # Save the model every epoch
        print(f"Saving Model for this epoch...")
        torch.save(op_model.state_dict(), f"nfnet_f1_model.pth")