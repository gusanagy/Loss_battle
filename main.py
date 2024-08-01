import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.perceptualLoss import *
from models import Unet_model, VAE_model, Vit_model
import torchinfo as t

h = PerceptualLoss(id=4)

print(h.name, h.id)

in_channels = 3  # Por exemplo, RGB
out_channels = 3  # Saída de imagem RGB
base_filters = 64
num_layers = 4
use_batch_norm = True
model = Unet_model(in_channels, out_channels, base_filters, num_layers, use_batch_norm)
x = torch.randn(1, 3, 256, 256)
print(model(x).shape,t.summary(model, input_size=(1, 3, 256, 256)))
latent_dim = 16
hidden_dims = [32, 64, 128, 256]
input_channels = 3
output_channels = 3
model = VAE_model(in_channels, out_channels, base_filters, num_layers, use_batch_norm)
print(model(x).shape,t.summary(model, input_size=(1, 3, 256, 256)))

## carregar modelos
## carregar funcoes de perda
## carrregar datasets
## Fazer combinacao de modelos e funcoes de perda
## implementar ddp e torchrun para treinamento distribuido e acelerado
## Treinar modelo 
## avaliar com as metricas
