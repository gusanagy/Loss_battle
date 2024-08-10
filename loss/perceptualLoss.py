####################
"""Perceptual Loss Functions in this file"""
####################

#Importing libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import mean, round, transpose
from time import time
import lpips
import torchvision.models as models
import cv2
import warnings
from typing import List

# Suprimir apenas FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
def list_perceptual_loss(list_loss: List[str] = None, rank = None):
    dict_loss = {
    'vgg11': PerceptualLoss(model='vgg11').to(rank),
    'vgg11_bn': PerceptualLoss(model='vgg11_bn').to(rank),
    'vgg13': PerceptualLoss(model='vgg13').to(rank),
    'vgg13_bn': PerceptualLoss(model='vgg13_bn').to(rank),
    'vgg16': PerceptualLoss(model='vgg16').to(rank),
    'vgg16_bn': PerceptualLoss(model='vgg16_bn').to(rank),
    'vgg19': PerceptualLoss(model='vgg19').to(rank),
    'vgg19_bn': PerceptualLoss(model='vgg19_bn').to(rank),
    'squeeze': PerceptualLoss(model='squeeze').to(rank),
    'alex': PerceptualLoss(model='alex').to(rank)    
                }
    return_loss =[]
    for loss in list_loss:
        if loss not in dict_loss.keys():
            raise ValueError(f"Unsupported perceptual model type\nPlease choose from {dict_loss.keys()}")
        return_loss.append(dict_loss[loss])
    return return_loss

def list_perceptual_loss_lazy(list_loss: List[str] = None, rank = None):
    dict_loss = {
        'vgg11': None,
        'vgg11_bn': None,
        'vgg13': None,
        'vgg13_bn': None,
        'vgg16': None,
        'vgg16_bn': None,
        'vgg19': None,
        'vgg19_bn': None,
        'squeeze': None,
        'alex': None
    }
    return_loss = []
    for loss in list_loss:
        if loss not in dict_loss.keys():
            raise ValueError(f"Unsupported perceptual model type\nPlease choose from {dict_loss.keys()}")
        if dict_loss[loss] is None:
            return_loss.append(PerceptualLoss(model=loss).to(rank))
    return return_loss


# Definindo a PerceptualLoss
class PerceptualLoss(nn.Module):
    def __init__(self, id: int = None, model='vgg16', layer_indices=None):
        super(PerceptualLoss, self).__init__()
        # Load the model
        self._id = id
        if model == 'vgg11':
            self.perceptual = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1).features
        elif model == 'vgg11_bn':
            self.perceptual = models.vgg11_bn(weights=models.VGG11_BN_Weights.IMAGENET1K_V1).features
        elif model == 'vgg13':
            self.perceptual = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1).features
        elif model == 'vgg13_bn':
            self.perceptual = models.vgg13_bn(weights=models.VGG13_BN_Weights.IMAGENET1K_V1).features
        elif model == 'vgg16':
            self.perceptual = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        elif model == 'vgg16_bn':
            self.perceptual = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1).features
        elif model == 'vgg19':
            self.perceptual = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        elif model == 'vgg19_bn':
            self.perceptual = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1).features
        elif model == 'squeeze':
            self.perceptual = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1).features
        elif model == 'alex':
            self.perceptual = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).features
        else:
            raise ValueError("Unsupported perceptual model type. Choose from ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'squeeze', 'alex']")
        
        self._model = model
        
        self.perceptual.eval()  # Set to evaluation mode
        for param in self.perceptual.parameters():
            param.requires_grad = False  # Freeze the parameters
        
        self.layer_indices = {
            'squeeze':  [3, 7, 12],
            'vgg11':    [3, 8, 15, 22],
            'vgg11_bn': [3, 8, 15, 22],
            'vgg13':    [3, 8, 15, 22],
            'vgg13_bn': [3, 8, 15, 22],
            'vgg16':    [3, 8, 15, 22],
            'vgg16_bn': [3, 8, 15, 22],
            'vgg19':    [3, 8, 17, 26, 35],
            'vgg19_bn': [3, 8, 17, 26, 35],
            'alex':     [3, 6, 8, 10, 12],
        }

        if layer_indices is not None:
            self.layer_indices[model] = layer_indices
    
    @property
    def name(self):
        return self.__class__.__name__ + '_' + self._model

    @property
    def id(self):
        return self._id
    
    def forward(self, x, y):
        # Normalize the inputs
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        y = (y - mean) / std

        # Extract features
        x_features = self.extract_features(x)
        y_features = self.extract_features(y)

        # Calculate perceptual loss
        loss = 0.0
        for xf, yf in zip(x_features, y_features):
            loss += nn.functional.l1_loss(xf, yf, reduction='mean')

        return loss

    def extract_features(self, x):
        features = []
        for i, layer in enumerate(self.perceptual):
            x = layer(x)
            if i in self.layer_indices[self._model]:
                features.append(x)
        return features





