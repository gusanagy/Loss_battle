import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.ChannelLoss import *

h = HistogramColorLoss(number=1)
print(h.number, h.name)

## carregar modelos
## carregar funcoes de perda
## carrregar datasets
## Fazer combinacao de modelos e funcoes de perda
## implementar ddp e torchrun para treinamento distribuido e acelerado
## Treinar modelo 
## avaliar com as metricas
