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

loss_squeeze = lpips.LPIPS(net='squeeze')
loss_alex = lpips.LPIPS(net='alex')


"""VGG Perceptual Loss"""
class VGGPerceptualLoss(nn.Module):
    def __init__(self, vgg_model='vgg16', layer_indices=None):
        super(VGGPerceptualLoss, self).__init__()
        # Load the VGG model
        if vgg_model == 'vgg11':
            self.vgg = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1).features
        elif vgg_model == 'vgg11_bn':
            self.vgg = models.vgg11_bn(weights=models.VGG11_BN_Weights.IMAGENET1K_V1).features
        elif vgg_model == 'vgg13':
            self.vgg = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1).features
        elif vgg_model == 'vgg13_bn':
            self.vgg = models.vgg13_bn(weights=models.VGG13_BN_Weights.IMAGENET1K_V1).features
        elif vgg_model == 'vgg16':
            self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        elif vgg_model == 'vgg16_bn':
            self.vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1).features
        elif vgg_model == 'vgg19':
            self.vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        elif vgg_model == 'vgg19_bn':
            self.vgg = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1).features
        else:
            raise ValueError("Unsupported VGG model type")

        self.vgg.eval()  # Set to evaluation mode
        for param in self.vgg.parameters():
            param.requires_grad = False  # Freeze the parameters

        # Specify the layers to extract features from
        if layer_indices is None:
            self.layer_indices = [3, 8, 15, 22]  # Default layers for VGG16
        else:
            self.layer_indices = layer_indices

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
            loss += nn.functional.l1_loss(xf, yf)

        return loss

    def extract_features(self, x):
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layer_indices:
                features.append(x)
        return features


"""Gradient Loss OpenCV"""
class GradientLossOpenCV(nn.Module):
    def __init__(self):
        super(GradientLossOpenCV, self).__init__()
    
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



