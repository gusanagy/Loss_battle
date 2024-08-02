
import torch.nn as nn
from .unet import *
from .vae import *
from .vit import *


def Unet_model(in_channels, out_channels, base_filters, num_layers, use_batch_norm):
    return UNet(in_channels, out_channels, base_filters, num_layers, use_batch_norm)
def VAE_model(input_channels, hidden_dims, latent_dim, output_channels):
    return VAE(input_channels, hidden_dims, latent_dim, output_channels)
def Vit_model(image_size=256, patch_size=16, num_channels=3, embed_dim=512, num_heads=8, mlp_dim=1024, num_layers=6, dropout=0.1):
    return ImageEnhancerTransformer(image_size=image_size, patch_size=patch_size, num_channels=num_channels, embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, num_layers=num_layers, dropout=dropout)