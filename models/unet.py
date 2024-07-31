#####################
"""U-Net Architecture implemented below"""
#####################

#Importing Libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as transforms
#from torchvision import transforms 
import numpy as np
from PIL import Image

class DoubleConv(nn.Module):
    """
    Bloco de duas convoluções 3x3 seguidas de ReLU.
    Permite a parametrização do uso de BatchNorm.
    """
    def __init__(self, in_channels, out_channels, use_batch_norm=True):
        super(DoubleConv, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.extend([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ])
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    """
    Implementação da arquitetura U-Net parametrizável para geração de imagens.
    Permite a parametrização do número de blocos de convolução e o uso de BatchNorm.
    """
    def __init__(self, in_channels, out_channels, base_filters=64, num_layers=5, use_batch_norm=True):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_filters = base_filters
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm

        # Encoder
        self.encoder_layers = nn.ModuleList()
        filters = base_filters
        
        for i in range(num_layers-1):
            self.encoder_layers.append(DoubleConv(in_channels, filters, use_batch_norm))
            in_channels = filters
            filters *= 2
        
        # Max-pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = DoubleConv(filters // 2, filters, use_batch_norm)

        # Decoder
        self.up_layers = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        
        
        for i in range(num_layers-1, 0, -1):#range(num_layers-1, 0, -1)
            filters //= 2
            self.up_layers.append(nn.ConvTranspose2d(filters * 2, filters, kernel_size=2, stride=2))
            self.up_convs.append(DoubleConv(filters * 2, filters , use_batch_norm))
            

        # Final convolution
        
        self.final_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)
    def see_model(self):
        print(f"""
        encoder: {self.encoder_layers}
        pool: {self.pool}
        botleneck: {self.bottleneck}
        up laeyers: {self.up_layers}
        up_convs: {self.up_convs}
        final conv: {self.final_conv}""")

    def forward(self, x):
        # Encoder
        encoder_results = []
        for layer in self.encoder_layers:
            x = layer(x)
            encoder_results.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i, (up, conv) in enumerate(zip(self.up_layers, self.up_convs)):
            x = up(x)
            x = torch.cat((x, encoder_results[-(i+1)]), dim=1)
            x = conv(x)

        return self.final_conv(x)
