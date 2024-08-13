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

def train_models(plot_epc:int = None,epochs: int=100, model_name=None, models: List[str] =None ,perceptual_loss: List[str] = ['vgg11', 'vgg16', 'vgg19','alex', 'squeeze'],channel_loss: List[str] = ['Histogram_loss','angular_color_loss', 'dark_channel_loss','lch_channel_loss','hsv_channel_loss'],structural_loss: List[str] = ['ssim', 'psnr', 'mse', 'gradientLoss'], dataset_name="UIEB", dataset_path="data"):
    
    ckpt_savedir, results_savedir, txt_savedir = check_dir()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modelos = load_models(models=models)
    loss_battle = []
    
    if perceptual_loss is not None:
        loss_battle.extend(build_perceptual_losses(perceptual_loss=perceptual_loss,rank=device))
    if channel_loss is not None:
        loss_battle.extend(build_channel_losses(channel_loss=channel_loss,rank = device))
    if structural_loss is not None:
         loss_battle.extend(build_structural_losses(structural_loss=structural_loss,rank = device))

    print(f"{len(loss_battle)} loss functions to train with")
    print(f"{len(modelos)} models to train with\n")

    # Dataloader UIEB
    train_loader_UIEB, test_loader_UIEB = create_dataloader(dataset_name=dataset_name, dataset_path=dataset_path,ddp=False)

    for model in modelos:
        for loss_fn in loss_battle:
            print(f"""Training: {model.__class__.__name__} with {loss_fn.name}""")
            model_name = model.__class__.__name__+'_'+ loss_fn.name
            model = model.to(device)  # Remova o rank
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

            for epoch in tqdm(range(epochs)):
                model.train()

                if model.__class__.__name__ == 'VAE':
                    for batch_idx, (data, target) in enumerate(train_loader_UIEB):
                        data, target = data.cuda(), target.cuda()
                        optimizer.zero_grad()
                        output ,mu, logvar = model(data)
                        loss = loss_fn(output, target)+loss_VAE(mu=mu, logvar=logvar)
                        loss = loss.requires_grad_(True)
                        loss.backward()
                        optimizer.step()
                        #Optionally print loss information
                        #print(f"Epoch [{epoch}/{epochs}], Batch [{batch_idx}/{len(train_loader_UIEB)}], Loss: {loss.item()} \n\n torch tensor:\n {output.shape,output} \n\n")
                        if batch_idx % plot_epc == 0:
                            print(f"Epoch [{epoch}/{epochs}], Batch [{batch_idx}/{len(train_loader_UIEB)}], Loss: {loss.item()} \n")
                else:
                    for batch_idx, (data, target) in enumerate(train_loader_UIEB):
                        #data_max, target_max, data_min, target_min = data.numpy().max(), target.numpy().max(), data.numpy().min(), target.numpy().min()
                        #print(f"Data: {data_max, data_min} \n\n Target: {target_max, target_min} \n\n")
                        data, target = data*2-1, target*2-1 ;"""quando fazer a iferencia precisa fazer essa transformacao e para plotar tem de fazer a transformcao inversa"""
                        data, target = data.cuda(), target.cuda()
                        #print(f"nova data:{new_data.numpy().max(), new_data.numpy().min()}")    
                        
                        optimizer.zero_grad()
                        output = model(data)
                        
                        loss = loss_fn(output, target).requires_grad_(True)

                        loss.backward()
                        optimizer.step()
                        
                        if epoch % plot_epc == 0:
                            print(f"Epoch [{epoch}/{epochs}], Batch [{batch_idx}/{len(train_loader_UIEB)}], Loss: {loss.item()}")


            # Salve Dir para salvar os checkpoints
            print(f"Testando o modelo {model_name}")
            # Salvar o estado do modelo original
            torch.save(model.state_dict(), f"{ckpt_savedir}{model_name}_ckpt.pth")
            # psnr_list, ssim_list, uciqe_list, uiqm_list = [], [], [], []
            # # Avaliar o modelo
            # model.eval()
            # with torch.no_grad():
            #     for batch_idx, (data, target) in tqdm(enumerate(test_loader_UIEB)):
            #         data, target = data.cuda(), target.cpu().numpy()
            #         predictions = model(data).cpu().numpy()
            #         # Calcula a métrica
            #         psnr_value, ssim_value, uciqe_, uiqm = calculate_metrics(predictions, target)
            #         psnr_list.append(psnr_value)
            #         ssim_list.append(ssim_value)
            #         uciqe_list.append(uciqe_)
            #         uiqm_list.append(uiqm)
            #         avg_ssim = sum(ssim_list) / len(ssim_list)
            # avg_psnr = sum(psnr_list) / len(psnr_list)
            # avg_uciqe = sum(uciqe_list) / len(uciqe_list)
            # avg_uiqm = sum(uiqm_list) / len(uiqm_list)
            
            # # Salvar métricas em um arquivo
            # with open(f'{results_savedir}{model_name}_metrics.txt', 'w') as f:
            #     f.write(f"""avg_ssim:{avg_ssim}\navg_psnr:{avg_psnr}\navg_uciqe:{avg_uciqe}\navg_uiqm:{avg_uiqm}""")
            #     print(f"Metrics for {model_name} saved to {results_savedir}/{model_name}_metrics.txt")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs per model")
    args = parser.parse_args()

    train_models(epochs=args.epochs)
