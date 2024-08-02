from .channelLoss import *
from .perceptualLoss import PerceptualLoss
from .structuralLoss import *
from typing import List
from tqdm.notebook import tqdm



def build_perceptual_losses(perceptual_loss: List[str] = ['vgg11', 'vgg16', 'vgg19','alex', 'squeeze']):
    PerceptualLosses = []
    id = 0
    for loss in tqdm(perceptual_loss):
       PerceptualLosses.append(PerceptualLoss(model=loss, id=id))
       id += 1
    return PerceptualLosses
def build_channel_losses(channel_loss: List[str] = ['Histogram_loss','angular_color_loss', 'dark_channel_loss','lch_channel_loss','hsv_channel_loss']):
    ChannelLosses = []
    list_channel_loss(channel_loss)
    id = 0
    for loss in tqdm(channel_loss):
        ChannelLosses.append((loss, id))
        id += 1
    return ChannelLosses
def build_structural_losses(structural_loss: List[str] = ['ssim', 'psnr', 'mse', 'gradientLoss']):
    structural = []
    list_structural_loss(structural_loss)
    id = 0
    for loss in tqdm(structural_loss):
       structural.append((loss, id))
       id += 1
    return structural