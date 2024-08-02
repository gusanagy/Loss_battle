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
from typing import List
# from torch.nn import MSELoss #L2LOSS


def list_structural_loss(list_loss: List[str] = None):
    dict_loss = {
        'ssim': SSIMLoss(),
        'psnr': PSNRLoss(),
        'mse': MSELoss(),
        'gradientLoss': GradientLossOpenCV(),
        'ms_ssim': MSSSIMLoss(),
        'charbonnier': CharbonnierLoss(),
        'l1': L1Loss(),
        'mae': MAELoss()
    }
    return_loss =[]
    for loss in list_loss:
        if loss not in dict_loss.keys():
            raise ValueError(f"Unsupported perceptual model type\nPlease choose from {dict_loss.keys()}")
        return_loss.append(dict_loss[loss])
    return return_loss
"""Mae Loss function"""
class MAELoss(nn.Module):
    def __init__(self, id: int = None):
        super(MAELoss, self).__init__()
        self.criterion = nn.L1Loss()
        self._id = id
    @property
    def name(self):
        return self.__class__.__name__
    @property
    def id(self):
        return self._id

    def forward(self, input, target):
        return self.criterion(input, target)
    
"""MSE Loss Function"""#%
class MSELoss(nn.Module):
    def __init__(self, id: int = None):
        super(MSELoss, self).__init__()
        self.criterion = nn.MSELoss()
        self._id = id
    @property
    def name(self):
        return self.__class__.__name__
    @property
    def id(self):
        return self._id
    def forward(self, input, target):
        return self.criterion(input, target)

"""L1 Loss Function"""
class L1Loss(nn.Module):
    def __init__(self, id: int = None):
        super(L1Loss, self).__init__()
        self._id = id
    @property
    def name(self):
        return self.__class__.__name__
    @property
    def id(self):
        return self._id

    def forward(self, input, target):
        # Calcula a diferença absoluta entre o input e o target
        abs_diff = torch.abs(input - target)
        # Calcula a média da diferença absoluta
        loss = torch.mean(abs_diff)
        return loss

"""Gradient Loss OpenCV"""#%
class GradientLossOpenCV(nn.Module):
    def __init__(self,id: int = None):
        super(GradientLossOpenCV, self).__init__()
        self._id = id
    @property
    def id(self):
        return self._id
    def name(self):
        return self.__class__.__name__
    
    def compute_gradient(self, image):
        """
        Compute the gradient of an image using OpenCV Sobel filters.

        Args:
            image (Tensor): The image with shape (N, C, H, W).

        Returns:
            Tensor: The gradient magnitude with shape (N, C, H, W).
        """
        # Convert PyTorch tensor to NumPy array
        image_np = image.detach().cpu().numpy()
        N, C, H, W = image_np.shape
        
        gradient_magnitude = np.zeros((N, C, H, W), dtype=np.float32)
        
        for n in range(N):
            for c in range(C):
                img = image_np[n, c, :, :]
                grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
                grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
                gradient_magnitude[n, c, :, :] = grad_mag
        
        # Convert NumPy array back to PyTorch tensor
        gradient_magnitude_tensor = torch.tensor(gradient_magnitude).to(image.device)
        
        return gradient_magnitude_tensor

    def forward(self, input, target):
        """
        Compute the gradient loss between the input and target images.

        Args:
            input (Tensor): The predicted image with shape (N, C, H, W).
            target (Tensor): The ground truth image with shape (N, C, H, W).

        Returns:
            Tensor: The Gradient Loss value.
        """
        # Compute gradients
        gradient_input = self.compute_gradient(input)
        gradient_target = self.compute_gradient(target)
        
        # Compute the loss
        loss = F.mse_loss(gradient_input, gradient_target)
        return loss

"""SSIM Loss function"""#%
class SSIMLoss(nn.Module):
    def __init__(self, id: int = None):
        super(SSIMLoss, self).__init__()
        self.criterion = ssim
        self._id = id

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def id(self):
        return self._id

    def forward(self, input, target):
        return self.criterion(input, target)

"""PSNR Loss function"""#%
class PSNRLoss(nn.Module):
    def __init__(self, id: int = None):
        super(PSNRLoss, self).__init__()
        self.criterion = psnr
        self._id = id

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def id(self):
        return self._id

    def forward(self, input, target):
        return self.criterion(input, target)

"""MS-SSIM Loss function"""
class MSSSIMLoss(nn.Module):
    def __init__(self, id: int = None):
        super(MSSSIMLoss, self).__init__()
        self.criterion = ms_ssim()
        self._id = id

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def id(self):
        return self._id

    def forward(self, input, target):
        return self.criterion(input, target)

"""Charbonnier Loss function"""
class CharbonnierLoss(nn.Module):
    def __init__(self, id: int = None):
        super(CharbonnierLoss, self).__init__()
        self.criterion = charbonnier
        self._id = id

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def id(self):
        return self._id

    def forward(self, input, target):
        return self.criterion(input, target)



