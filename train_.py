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

def check_save_dir(dir:str = None):
    """
    Cria pastas e subpastas para salvar os resultados.
    Pasta:
        output/ckpt_study/(dataset + model + loss)
        
    """
    filter = dir.split('.')
    if dir is None:
        print("Por favor, insira um nome para o diretório")
        return 0
    else:
        ckpt_savedir = filter[0] +'/'
        if not os.path.exists(ckpt_savedir):
            os.makedirs(ckpt_savedir)
    return ckpt_savedir

def train_final(plot_epc:int = 700,epochs: int=100, model_name=None, 
                perceptual_loss: str = None, structural_loss: str= None,
                channel_loss: str = None, pretrained: str = None,
                dataset_name="UIEB", dataset_path="data",
                ckpt_out_name: str = None):
    
    ckpt_savedir, results_savedir, txt_savedir = check_dir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_one_model(model=model_name)
    if pretrained is not None:
        # Carregar os pesos do arquivo
        state_dict = torch.load(pretrained, map_location=device)
        
        # Carregar os pesos no modelo
        model.load_state_dict(state_dict)

    loss_l = []
    if perceptual_loss is not None:
        loss_l.append(load_perceptual_loss(list_loss=perceptual_loss,rank=device))
    if channel_loss is not None:
        loss_l.append(load_channel_loss(list_loss=channel_loss,rank = device))
    if structural_loss is not None:
        loss_l.append(load_structural_loss(list_loss=structural_loss,rank = device))
            #print(f"{loss}")

    Loss_unet_vit = nn.MSELoss()

    # Dataloader UIEB
    train_loader_UIEB, test_loader_UIEB = create_dataloader(dataset_name=dataset_name, dataset_path=dataset_path, ddp=False, batch_size=1)

    # print(f"""Training: {model.__class__.__name__} with {loss_fn.name}""")
    # model_name = model.__class__.__name__+'_'+ loss_fn.name
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
    # Defina um valor para o clipping
    max_norm = 1.0
    for epoch in tqdm(range(epochs)):
        model.train()

        if model_name == 'VAE':
            for batch_idx, (data, target) in enumerate(train_loader_UIEB):
                data, target = data.cuda(), target.cuda()
                #data = data.to(device)
                
                optimizer.zero_grad()
                #output = model(data)
                output, mu, logvar = model(data)
                
                # Calculando a perda total #loss = loss_1(output, target) + loss_2(output, target) + loss_3(output, target)
                loss = loss_VAE(data, output, mu, logvar)
                for l in loss_l:
                    loss += l(output, target)
                
                loss = loss.requires_grad_(True)

                
                loss.backward()
                # Adicionando clipping de gradientes
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                if batch_idx % plot_epc == 0:
                    print(f"Epoch [{epoch}/{epochs}], Batch [{batch_idx}/{len(train_loader_UIEB)}], Loss: {loss.item()} \n")
            
        elif model_name == 'Unet' or model_name == 'Vit':
            for batch_idx, (data, target) in enumerate(train_loader_UIEB):
                data, target = data.cuda(), target.cuda()
                
                optimizer.zero_grad()
                output = model(data)

                loss = Loss_unet_vit(output,target)
                # Calculando a perda total               
                for l in loss_l:
                    loss += l(output, target)
                loss = loss.requires_grad_(True)
            
                loss.backward()
                # Adicionando clipping de gradientes
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                
                if batch_idx % plot_epc == 0:
                    print(f"Epoch [{epoch}/{epochs}], Batch [{batch_idx}/{len(train_loader_UIEB)}], Loss: {loss.item()} \n")
            
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target, name) in enumerate(test_loader_UIEB):
            if batch_idx == 1:
                break
            data, target = data.cuda(), target.cuda()
            if model_name == 'VAE':
                output, mu, logvar = model(data)
            else:
                output = model(data)
            print(f"Epoch [{epoch}/{epochs}], Batch [{batch_idx}/{len(test_loader_UIEB)}]")
            for i in output.cpu().numpy().transpose(0, 2, 3, 1):
                plt.imshow(i)
                plt.show()

    # Salve Dir para salvar os checkpoints
    name = f'{model_name}_{dataset_name}_{loss_l[0].name}_{epochs}'
    print(f"Salvando Ckpt {name}")
    # Salvar o estado do modelo original
    if ckpt_out_name is not None:   
        torch.save(model.state_dict(), f"{ckpt_savedir}{name}_{ckpt_out_name}.pth")
        check_save_dir(dir = f'{ckpt_savedir}{name}_{ckpt_out_name}')
    elif perceptual_loss is None and structural_loss is None and channel_loss is None:
        torch.save(model.state_dict(), f"{ckpt_savedir}{model_name}_{dataset_name}_PRETRAINED_{epochs}.pth")
        check_save_dir(dir = f'{ckpt_savedir}{model_name}_{dataset_name}_PRETRAINED_{epochs}')
    else:
        torch.save(model.state_dict(), f"{ckpt_savedir}{name}.pth")
        check_save_dir(dir = f'{ckpt_savedir}{name}')
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs per model")
    args = parser.parse_args()
    train_final(epochs=args.epochs)

