import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from loss import *
from models import *
from src.dataload import *
from metrics.metrics import *
from tqdm import tqdm  # Use tqdm para ambientes locais, não notebook
import cv2

def test_one_model(model_name='Unet', dataset_name="UIEB", dataset_path="data", ckpt_path=None):
    #"output/ckpt_battle/UNet_PerceptualLoss_vgg11_ckpt.pth"
    filename = ckpt_path.split('/')[-1].split('.')[0]
    results_savedir = f'output/results_battle/{filename}/'
    if not os.path.exists(results_savedir):
        os.makedirs(results_savedir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_one_model(model=model_name)
    
    loss_test = build_perceptual_losses(perceptual_loss=['vgg16'],rank=device)
    # loss_battle.extend(build_perceptual_losses(rank=device))
    # loss_battle.extend(build_channel_losses(rank = device))
    # loss_battle.extend(build_structural_losses(rank = device))

    print(f"{len(loss_test)} loss functions to train with")
    print(f"{len(model)} models to train with\n")

    # Dataloader UIEB
    train_loader_UIEB, test_loader_UIEB = create_dataloader(dataset_name=dataset_name, dataset_path=dataset_path,ddp=False)
    # Salve Dir para salvar os checkpoints
    print(f"Testando o modelo {model_name}")
    # Salvar o estado do modelo original
    model.load_state_dict(torch.load(PATH))
    torch.load(model.state_dict(), f"{ckpt_savedir}{model_name}_ckpt.pth")
    psnr_list, ssim_list, uciqe_list, uiqm_list = [], [], [], []
    
    # Avaliar o modelo
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(test_loader_UIEB)):
            data, target = data.cuda(), target.cpu().numpy()
            predictions = model(data).cpu().numpy()
            print(predictions.shape)
            #.detach().cpu().numpy()[0].transpose(1, 2, 0),0,1)[:,:,::-1]
            cv2.imwrite(f"{results_savedir}{model_name}_prediction.png", predictions*255)

            # Calcula a métrica
            psnr_value, ssim_value, uciqe_, uiqm = calculate_metrics(predictions, target)
            psnr_list.append(psnr_value)
            ssim_list.append(ssim_value)
            uciqe_list.append(uciqe_)
            uiqm_list.append(uiqm)
        avg_ssim = sum(ssim_list) / len(ssim_list)
        avg_psnr = sum(psnr_list) / len(psnr_list)
        avg_uciqe = sum(uciqe_list) / len(uciqe_list)
        avg_uiqm = sum(uiqm_list) / len(uiqm_list)

        
            
        # Salvar métricas em um arquivo
        with open(f'{results_savedir}{model_name}_metrics.txt', 'w') as f:
            f.write(f"""avg_ssim:{avg_ssim}\navg_psnr:{avg_psnr}\navg_uciqe:{avg_uciqe}\navg_uiqm:{avg_uiqm}""")
        print(f"Metrics for {model_name} saved to {results_savedir}/{model_name}_metrics.txt")

    pass



def test_models(epochs: int=100, loss_fn=None, model_name=None, model=None, dataset_name="UIEB", dataset_path="data"):
    ckpt_savedir = 'output/ckpt_battle/'
    if not os.path.exists(ckpt_savedir):
        os.makedirs(ckpt_savedir)
    results_savedir = 'output/results_battle/'
    if not os.path.exists(results_savedir):
        os.makedirs(results_savedir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelos = load_models()
    loss_battle = []

    loss_battle.extend(build_perceptual_losses(rank=device))
    loss_battle.extend(build_channel_losses(rank = device))
    loss_battle.extend(build_structural_losses(rank = device))
    

    print(f"{len(loss_battle)} loss functions to train with")
    print(f"{len(modelos)} models to train with\n")

    # Dataloader UIEB
    train_loader_UIEB, test_loader_UIEB = create_dataloader(dataset_name=dataset_name, dataset_path=dataset_path,ddp=False)



    model.eval()
            with torch.no_grad():
                for batch_idx, (data, target) in tqdm(enumerate(test_loader_UIEB)):
                    data, target = data.cuda(), target.cpu().numpy()
                    predictions = model(data).cpu().numpy()
                    # Calcula a métrica
                    psnr_value, ssim_value, uciqe_, uiqm = calculate_metrics(predictions, target)
                    psnr_list.append(psnr_value)
                    ssim_list.append(ssim_value)
                    uciqe_list.append(uciqe_)
                    uiqm_list.append(uiqm)̂
            avg_ssim = sum(ssim_list) / len(ssim_list)
            avg_psnr = sum(psnr_list) / len(psnr_list)
            avg_uciqe = sum(uciqe_list) / len(uciqe_list)
            avg_uiqm = sum(uiqm_list) / len(uiqm_list)
            
            # Salvar métricas em um arquivo
            with open(f'{results_savedir}{model_name}_metrics.txt', 'w') as f:
                f.write(f"""avg_ssim:{avg_ssim}\navg_psnr:{avg_psnr}\navg_uciqe:{avg_uciqe}\navg_uiqm:{avg_uiqm}""")
                print(f"Metrics for {model_name} saved to {results_savedir}/{model_name}_metrics.txt")

    pass