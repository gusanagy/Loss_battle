#####################
"""ViT Architecture implemented below"""
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
import torch.optim as optim

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        mlp_output = self.mlp(x)
        x = x + self.dropout(mlp_output)
        x = self.norm2(x)
        return x


class ImageEnhancerTransformer(nn.Module):
    def __init__(self, image_size=256, patch_size=16, num_channels=3, embed_dim=512, num_heads=8, mlp_dim=1024, num_layers=6, dropout=0.1):
        super(ImageEnhancerTransformer, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = num_channels * patch_size * patch_size
        self.embed_dim = embed_dim

        self.patch_embeddings = nn.Linear(self.patch_dim, embed_dim)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, self.patch_dim)
    @property
    def name(self):
        return self.__class__.__name__
    def forward(self, x):
        batch_size = x.size(0)
        x = self._to_patches(x)
        x = self.patch_embeddings(x) + self.position_embeddings

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        x = self.fc(x)
        x = self._from_patches(x, batch_size)
        return x

    def _to_patches(self, x):
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, self.image_size // self.patch_size, self.patch_size,
                   self.image_size // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(batch_size, self.num_patches, -1)
        return x

    def _from_patches(self, x, batch_size):
        x = x.view(batch_size, self.image_size // self.patch_size, self.image_size // self.patch_size,
                   self.patch_size, self.patch_size, -1)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(batch_size, -1, self.image_size, self.image_size)
        return x
