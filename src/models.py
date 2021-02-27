import numpy as np
import torch
import torch.nn as nn
import timm

class NFNetModel(nn.Module):
    """
    Model Class for the newly introduced Normalization Free Network (NFNet) Model Architecture
    """
    def __init__(self, num_classes=11, model_name='nfnet_f1', pretrained=True):
        super(NFNetModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.head.fc = nn.Linear(self.model.head.fc.in_features, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x