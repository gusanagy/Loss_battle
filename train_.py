import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from loss import *
from models import *
from src.dataload import *
from metrics.metrics import *
from src.utils import *
from tqdm import tqdm  # Use tqdm para ambientes locais, não notebook
import matplotlib.pyplot as plt

def train_final(plot_epc:int = 700,epochs: int=100, model_name=None, 
                pretrained = None,
                perceptual_loss: List[str] = ['vgg11', 'vgg16', 'vgg19','alex', 'squeeze'],
                channel_loss: List[str] = ['Histogram_loss','angular_color_loss', 'dark_channel_loss','lab_channel_loss','hsv_channel_loss'],
                structural_loss: List[str] = ['ssim', 'psnr', 'mse', 'gradientLoss'],
                dataset_name="UIEB", dataset_path="data"):
    
    ckpt_savedir, results_savedir, txt_savedir = check_dir()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_one_model(model=model_name)
    if pretrained is not None:
        # Carregar os pesos do arquivo
        state_dict = torch.load(pretrained, map_location=device)
        
        # Carregar os pesos no modelo
        model.load_state_dict(state_dict)
    #loss_1 = SSIMLoss()
    loss_1 = PerceptualLoss(model='vgg19').to(device)
    #loss_3 = DarkChannelLoss()

    # Dataloader UIEB
    train_loader_UIEB, test_loader_UIEB = create_dataloader(dataset_name=dataset_name, dataset_path=dataset_path,ddp=False)
       
    # print(f"""Training: {model.__class__.__name__} with {loss_fn.name}""")
    # model_name = model.__class__.__name__+'_'+ loss_fn.name
    model = model.to(device)  # Remova o rank
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
    # Defina um valor para o clipping
    max_norm = 1.0
    for epoch in tqdm(range(epochs)):
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader_UIEB):
            data, target = data.cuda(), target.cuda()
            
            optimizer.zero_grad()
            output = model(data)
            
            loss = loss_1(output, target) 
            loss = loss.requires_grad_(True)
          

            loss.backward()

            # Clip gradientes
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            
            if batch_idx % plot_epc == 0 :
                print(f"Epoch [{epoch}/{epochs}], Batch [{batch_idx}/{len(train_loader_UIEB)}], Loss: {loss.item()} \n")
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target, name) in enumerate(test_loader_UIEB):
            if batch_idx == 1:
                break
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = loss_1(output, target)
            print(f"Epoch [{epoch}/{epochs}], Batch [{batch_idx}/{len(test_loader_UIEB)}], Loss: {loss.item}")
            for i in output.cpu().numpy().transpose(0, 2, 3, 1):
                plt.imshow(i)
                plt.show()
    # Salve Dir para salvar os checkpoints
    print(f"Salvando Ckpt {model_name}")
    # Salvar o estado do modelo original
    torch.save(model.state_dict(), f"{ckpt_savedir}{model_name}_{dataset_name}_ckpt.pth")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs per model")
    args = parser.parse_args()

    train_final(epochs=args.epochs)
