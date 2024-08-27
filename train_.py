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

def train_final(plot_epc:int = 700,epochs: int=100, model_name=None, 
                pretrained = None,
                perceptual_loss: List[str] = ['vgg11', 'vgg16', 'vgg19','alex', 'squeeze'],
                channel_loss: List[str] = ['Histogram_loss','angular_color_loss', 'dark_channel_loss','lab_channel_loss','hsv_channel_loss'],
                structural_loss: List[str] = ['ssim', 'psnr', 'mse', 'gradientLoss'], dataset_name="UIEB", dataset_path="data"):
    
    ckpt_savedir, results_savedir, txt_savedir = check_dir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_one_model(model=model_name)

    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained), strict=False)
    loss_1 = SSIMLoss()
    loss_2 = PerceptualLoss(model='vgg19').to(device)

    # Dataloader UIEB
    train_loader_UIEB, test_loader_UIEB = create_dataloader(dataset_name=dataset_name, dataset_path=dataset_path, ddp=False, batch_size=1)

    # print(f"""Training: {model.__class__.__name__} with {loss_fn.name}""")
    # model_name = model.__class__.__name__+'_'+ loss_fn.name
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

    for epoch in tqdm(range(epochs)):
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader_UIEB):
            data, target = data.cuda(), target.cuda()
            #data = data.to(device)
            
            optimizer.zero_grad()
            #output = model(data)
            output, mu, logvar = model(data)
            
            #loss = loss_1(output, target) + loss_2(output, target) + loss_3(output, target)
            #loss = loss.requires_grad_(True)
            # Calculando a perda total
            loss = loss_VAE(data, output, mu, logvar) #+ loss_1(output, target)
            
            loss.backward()
            # Adicionando clipping de gradientes
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if batch_idx % plot_epc == 0:
                print(f"Epoch [{epoch}/{epochs}], Batch [{batch_idx}/{len(train_loader_UIEB)}], Loss: {loss.item()} \n")

    # Salve Dir para salvar os checkpoints
    print(f"Salvando Ckpt {model_name}")
    # Salvar o estado do modelo original
    torch.save(model.state_dict(), f"{ckpt_savedir}{model_name}_{dataset_name}_final_ckpt.pth")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs per model")
    args = parser.parse_args()
    train_final(epochs=args.epochs)

