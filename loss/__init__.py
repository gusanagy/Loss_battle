from .channelLoss import *
from .perceptualLoss import *
from .structuralLoss import *
from typing import List
from tqdm.notebook import tqdm

def build_perceptual_losses(perceptual_loss: List[str] = ['vgg11', 'vgg16', 'vgg19','alex', 'squeeze'],rank = None):
    return list_perceptual_loss(list_loss=perceptual_loss,rank=rank)
def build_channel_losses(channel_loss: List[str] = ['Histogram_loss','angular_color_loss', 'dark_channel_loss','lab_channel_loss','hsv_channel_loss'],rank = None):
    return list_channel_loss(channel_loss)
def build_structural_losses(structural_loss: List[str] = ['ssim', 'psnr', 'mse', 'gradientLoss'],rank = None):
    return list_structural_loss(structural_loss)

# Função de perda para VAE com perda perceptual
def loss_VAE(x, x_hat, mu, logvar):
    # Perda de reconstrução (Binary Cross-Entropy)
    recon_loss = F.mse_loss(x_hat, x, reduction='sum')
    #recon_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    # Perda KL-divergence #mede quão distante esta a distribuição aprendida 
    #do que queremos que a distribuição se pareça
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Soma das duas perdas
    total_loss = recon_loss + kl_loss
    return total_loss

#def loss_VAE(x, x_hat, mu, logvar, recon_weight=0.4, kl_weight=0.8):
    # Perda de reconstrução (Mean Squared Error ou Binary Cross-Entropy)
    #recon_loss = F.mse_loss(x_hat, x, reduction='sum')
    
    # Perda KL-divergence
    #kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Aplicar pesos às perdas
    #weighted_recon_loss = recon_weight * recon_loss
    #weighted_kl_loss = kl_weight * kl_loss
    
    # Soma das perdas ponderadas
    #total_loss = weighted_recon_loss + weighted_kl_loss
    
    #return total_loss



#def loss_VAE(x, x_hat, mu, logvar):
    # Perda de reconstrução (Binary Cross-Entropy)
    #recon_loss = F.binary_cross_entropy(x_hat, x, reduction='none')  # Usa 'none' para não reduzir por enquanto
    #recon_loss = recon_loss.view(recon_loss.size(0), -1)  # Achata a imagem
    #recon_loss = recon_loss.sum(dim=1)  # Soma a perda por pixel
    #recon_loss = recon_loss.mean()  # Calcula a média da perda de reconstrução

    # Perda KL-divergence
    #kl_loss = 1 + logvar - mu.pow(2) - logvar.exp()
    #kl_loss = kl_loss.sum(dim=1)  # Soma a perda KL por imagem
    #kl_loss = -0.5 * kl_loss.mean()  # Calcula a média da perda KL

    # Soma das duas perdas
    #total_loss = recon_loss + kl_loss
    #return total_loss