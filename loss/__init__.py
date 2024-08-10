from .channelLoss import *
from .perceptualLoss import *
from .structuralLoss import *
from typing import List
from tqdm.notebook import tqdm

def build_perceptual_losses(perceptual_loss: List[str] = ['vgg11', 'vgg16', 'vgg19','alex', 'squeeze'],rank = None):
    return list_perceptual_loss(list_loss=perceptual_loss,rank=rank)
def build_channel_losses(channel_loss: List[str] = ['Histogram_loss','angular_color_loss', 'dark_channel_loss','lch_channel_loss','hsv_channel_loss'],rank = None):
    return list_channel_loss(channel_loss)
def build_structural_losses(structural_loss: List[str] = ['ssim', 'psnr', 'mse', 'gradientLoss'],rank = None):
    return list_structural_loss(structural_loss)

# Função de perda para VAE com perda perceptual
def loss_VAE(mu, logvar):
    
    # Perda KL-divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return  kl_loss