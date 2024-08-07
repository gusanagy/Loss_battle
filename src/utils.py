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
from metrics import *

def test_one_model(model_name='Unet', dataset_name="UIEB", dataset_path="data", ckpt_path=None):
    #"output/ckpt_battle/UNet_PerceptualLoss_vgg11_ckpt.pth"
    filename = ckpt_path.split('/')[-1].split('.')[0]
    results_savedir = f'output/results_battle/{filename}/'
    if not os.path.exists(results_savedir):
        os.makedirs(results_savedir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_one_model(model=model_name).to(device)
    
    loss_test = build_perceptual_losses(perceptual_loss=['vgg16'],rank=device)
    # loss_battle.extend(build_perceptual_losses(rank=device))
    # loss_battle.extend(build_channel_losses(rank = device))
    # loss_battle.extend(build_structural_losses(rank = device))

    # Dataloader UIEB
    train_loader_UIEB, test_loader_UIEB = create_dataloader(dataset_name=dataset_name, dataset_path=dataset_path,ddp=False)
    # Salve Dir para salvar os checkpoints
    print(f"Testando o modelo {model_name}")
    # Salvar o estado do modelo original
    model.load_state_dict(torch.load(ckpt_path))
    psnr_list, ssim_list, uciqe_list, uiqm_list = [], [], [], []
    
    # Avaliar o modelo
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(test_loader_UIEB)):
            data = data.cuda()
            target = target.cpu().numpy().transpose(0, 2, 3, 1) # Convertendo para NHWC

            predictions = model(data).cpu().numpy().transpose(0, 2, 3, 1)  # Convertendo para NHWC
            print(f"pred.shape: {model(data)[:1]}")
            for i in range(predictions.shape[0]):
                pred_img = predictions[i][::-1]
                target_img = target[i][::-1]
                # print(f"""
                # pred_img.shape: {pred_img.shape}
                # data: {target_img.shape}
                # print: {pred_img.max(),p
                # """)
                # Normalizar se necessário (0-1)
                if pred_img.max() > 1.0:
                    pred_img = pred_img / 255.0
                if target_img.max() > 1.0:
                    target_img = target_img / 255.0
                plt.imshow(target_img)
                plt.show()
                plt.imshow(pred_img)
                plt.show()
                #Salvar a imagem predita
                #cv2.imwrite(f"{results_savedir}{model_name}_prediction_{batch_idx}_{i}.png", pred_img * 255)

                # Calcula a métrica
                psnr_value, ssim_value, uciqe_, uiqm = calculate_metrics(pred_img, target_img)
                print(f"PSNR: {psnr_value}, SSIM: {ssim_value}, UCIQE: {uciqe_}, UIQM: {uiqm}")
            break
        #         psnr_list.append(psnr_value)
        #         ssim_list.append(ssim_value)
        #         uciqe_list.append(uciqe_)
        #         uiqm_list.append(uiqm)
        # avg_ssim = sum(ssim_list) / len(ssim_list)
        # avg_psnr = sum(psnr_list) / len(psnr_list)
        # avg_uciqe = sum(uciqe_list) / len(uciqe_list)
        # avg_uiqm = sum(uiqm_list) / len(uiqm_list)

        
            
        # # Salvar métricas em um arquivo
        # with open(f'{results_savedir}{model_name}_metrics.txt', 'w') as f:
        #     f.write(f"""avg_ssim:{avg_ssim}\navg_psnr:{avg_psnr}\navg_uciqe:{avg_uciqe}\navg_uiqm:{avg_uiqm}""")
        # print(f"Metrics for {model_name} saved to {results_savedir}/{model_name}_metrics.txt")

    pass
def calculate_metrics(pred_img, target_img):
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