
import torch.nn as nn
from .unet import *
from .vae import *
from .vit import *


def Unet_model(in_channels, out_channels, base_filters, num_layers, use_batch_norm):
    return UNet(in_channels, out_channels, base_filters, num_layers, use_batch_norm)
def VAE_model(in_channels, out_channels, base_filters, num_layers, use_batch_norm):
    return VAE(in_channels, out_channels, base_filters, num_layers, use_batch_norm)
def Vit_model(in_channels, out_channels, base_filters, num_layers, use_batch_norm):
    return ImageEnhancerTransformer(in_channels, out_channels, base_filters, num_layers, use_batch_norm)