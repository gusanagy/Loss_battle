
import torch.nn as nn
from .unet import *
from .vae import *
from .vit import *
from typing import List

def Unet_model(in_channels:int = 3, out_channels:int = 3, base_filters:int = 64, num_layers:int = 4, use_batch_norm:bool = True):
    return UNet(in_channels, out_channels, base_filters, num_layers, use_batch_norm)
def VAE_model(input_channels:int = 3, hidden_dims: List[int] = [32, 64, 128, 256], latent_dim:int = 16, output_channels: int= 3):
    return VAE(input_channels, hidden_dims, latent_dim, output_channels)
def Vit_model(image_size:int=256, patch_size:int=16, num_channels:int=3, embed_dim:int=512, num_heads:int=8, mlp_dim:int=1024, num_layers:int=6, dropout:int=0.1):
    return ImageEnhancerTransformer(image_size=image_size, patch_size=patch_size, num_channels=num_channels, embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, num_layers=num_layers, dropout=dropout)