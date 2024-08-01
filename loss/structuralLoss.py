####################
"""Structural Loss Functions in this file"""
####################

#Importing libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.losses import ssim_loss as ssim, psnr_loss as psnr, MS_SSIMLoss as ms_ssim, charbonnier_loss as charbonnier
import numpy as np
from numpy import mean, round, transpose
from time import time
import lpips
import torchvision.models as models
import cv2
from torch.nn import MSELoss #L2LOSS


"""Mae Loss function"""
class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, input, target):
        return self.criterion(input, target)
    

"""MSE Loss Function"""
class MSELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, input, target):
        return self.criterion(input, target)



"""L1 Loss Function"""
class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, input, target):
        # Calcula a diferença absoluta entre o input e o target
        abs_diff = torch.abs(input - target)
        # Calcula a média da diferença absoluta
        loss = torch.mean(abs_diff)
        return loss





