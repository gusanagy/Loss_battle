import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import build_perceptual_losses, build_channel_losses
from models import Unet_model, VAE_model, Vit_model
import torchinfo as t


## carregar funcoes de perda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
per = build_perceptual_losses(['vgg11', 'vgg19'])

for i in per:
    print(i.name,i.id)

## carregar modelos


# in_channels = 3  # Por exemplo, RGB
# out_channels = 3  # Saída de imagem RGB
# base_filters = 64
# num_layers = 4
# use_batch_norm = True
# model = Unet_model(in_channels, out_channels, base_filters, num_layers, use_batch_norm)
x = torch.randn(1, 3, 256, 256)
# print(model(x).shape,t.summary(model, input_size=(1, 3, 256, 256)))
# Configuração de parâmetros
# latent_dim = 16
# hidden_dims = [32, 64, 128, 256]
# input_channels = 3
# output_channels = 3
# model = VAE_model(input_channels=input_channels, hidden_dims=hidden_dims, latent_dim=latent_dim, output_channels=output_channels).to(device)
# print(x.shape,t.summary(model, input_size=(1, 3, 256, 256)))

# model = Vit_model(image_size=256, patch_size=16, num_channels=3, embed_dim=512, num_heads=8, mlp_dim=1024, num_layers=6, dropout=0.1)

# print(x.shape,t.summary(model, input_size=(1, 3, 256, 256)))

## carregar datasets

## Fazer combinacao de modelos e funcoes de perda

## implementar ddp e torchrun para treinamento distribuido e acelerado

## Treinar modelo 

## avaliar com as metricas // colocar metricas de validaçao
