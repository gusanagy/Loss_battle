from .channelLoss import *
from .perceptualLoss import PerceptualLoss
from .structuralLoss import *
from typing import List



def build_perceptual_losses(perceptual_loss: List[str]):
    PerceptualLosses = []
    id = 0
    for loss in perceptual_loss:
       print(loss)
       PerceptualLosses.append(PerceptualLoss(model=loss, id=id))
       id += 1
    return PerceptualLosses
def build_channel_losses(channel_loss: List[str]):
    ChannelLosses = []
    id = 0
    for loss in channel_loss:
       ChannelLosses.append((loss, id))
       id += 1
    return ChannelLosses

def list_losses():
    
    return ['channelLoss', 'perceptualLoss', 'structuralLoss']

class list_loss(nn.Module):
    def __init__(self, loss_parameters, loss_group, loss_number):#receber a lista de parametros para as funcoes de perda
        super(list_loss, self).__init__()


        ### Inicializar todas as losses com seus respectivos parametros