import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.perceptualLoss import *

h = PerceptualLoss(id=4)
print(h.name, h.id)

## carregar modelos
## carregar funcoes de perda
## carrregar datasets
## Fazer combinacao de modelos e funcoes de perda
## implementar ddp e torchrun para treinamento distribuido e acelerado
## Treinar modelo 
## avaliar com as metricas
