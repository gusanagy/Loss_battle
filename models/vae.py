#####################
"""VAE Architecture implemented below"""
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

# Encoder parametrizável com BatchNorm
class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_dims, latent_dim):
        super(Encoder, self).__init__()
        
        # Inicializa uma lista para armazenar as camadas da rede
        modules = []
        
        # Adiciona camadas de convolução e batch normalization conforme especificado em hidden_dims
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU()
                )
            )
            input_channels = h_dim
        
        # Define as camadas finais para calcular mu e logvar
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 16 * 16, latent_dim)  # Supondo entrada de 256x256
        self.fc_logvar = nn.Linear(hidden_dims[-1] * 16 * 16, latent_dim)
    
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# Decoder parametrizável com BatchNorm
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_channels):
        super(Decoder, self).__init__()
        
        # Inicializa uma lista para armazenar as camadas da rede
        modules = []
        
        # Primeira camada completamente conectada para expandir a representação latente
        self.fc = nn.Linear(latent_dim, hidden_dims[-1] * 16 * 16)
        
        # Adiciona camadas de convolução transposta e batch normalization conforme especificado em hidden_dims
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=4,
                                       stride=2,
                                       padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.ReLU()
                )
            )
        
        # Camada final para reconstrução da imagem
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1],
                                   output_channels,
                                   kernel_size=4,
                                   stride=2,
                                   padding=1),
                nn.Sigmoid()
            )
        )
        
        self.decoder = nn.Sequential(*modules)
    
    def forward(self, z):
        x = torch.relu(self.fc(z))
        x = x.view(x.size(0), -1, 16, 16)
        x = self.decoder(x)
        return x

# Definição do VAE parametrizável
class VAE(nn.Module):
    def __init__(self, input_channels, hidden_dims, latent_dim, output_channels):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_channels, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, output_channels)
    @property
    def name(self):
        return self.__class__.__name__
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
    