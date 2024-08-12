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
    results_savedir = f'output/results_battle/{filename}/'#modificar devido a estrutura de pastas
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
            #print(f"pred.shape: {model(data)[:1]}")
            for i in range(predictions.shape[0]):
                if i == 4:
                    break
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
                plt.imshow(pred_img*255)
                plt.show()
                #Salvar a imagem predita
                #cv2.imwrite(f"{results_savedir}{model_name}_prediction_{batch_idx}_{i}.png", pred_img * 255)

                # Calcula a métrica
                #psnr_value, ssim_value, uciqe_, uiqm = calculate_metrics(pred_img, target_img)
                #print(f"PSNR: {psnr_value}, SSIM: {ssim_value}, UCIQE: {uciqe_}, UIQM: {uiqm}")
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

def calculate_metrics(pred_img, target_img):
    pass

def check_dir():
    ckpt_savedir = 'output/ckpt_battle/'
    if not os.path.exists(ckpt_savedir):
        os.makedirs(ckpt_savedir)
    results_savedir = 'output/results_battle/'
    if not os.path.exists(results_savedir):
        os.makedirs(results_savedir)
    txt_savedir = 'output/metrics_battle/'
    if not os.path.exists(txt_savedir):
        os.makedirs(txt_savedir)
    return ckpt_savedir, results_savedir, txt_savedir


def test_models(epochs: int=100, loss_fn=None, model_name=None, model=None, dataset_name="UIEB", dataset_path="data"):
    #check directories
    ckpt_savedir, results_savedir, txt_savedir = check_dir()
    #Agnostic code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Load Model
    modelos = load_models()
    #Append all losses
    loss_battle = []
    loss_battle.extend(build_perceptual_losses(rank=device))
    loss_battle.extend(build_channel_losses(rank = device))
    loss_battle.extend(build_structural_losses(rank = device))
    #Data for metrics
    psnr_list, ssim_list, uciqe_list, uiqm_list,loss_list = [], [], [], [], []

    print(f"{len(loss_battle)} loss functions to test with")
    print(f"{len(modelos)} models to test with\n")

    # Dataloader UIEB
    train_loader_UIEB, test_loader_UIEB = create_dataloader(dataset_name=dataset_name, dataset_path=dataset_path,ddp=False)

    for model in modelos:
        for loss_fn in loss_battle:
            print(f"""Testing: {model.__class__.__name__} with {loss_fn.name}""")
            model_name = model.__class__.__name__+'_'+ loss_fn.name

            model = model.to(device)  # Remova o rank

            model.load_state_dict(torch.load(f"{ckpt_savedir}{model_name}_ckpt.pth"))

            
            model.eval()
            with torch.no_grad():
                    
                if model.__class__.__name__ == 'VAE':
                    for batch_idx, (data, target) in tqdm(enumerate(test_loader_UIEB)):
                        if batch_idx == 1:
                            break
                        data, target = data.cuda(), target.cuda()
                            
                        output ,mu, logvar = model(data)
                        loss = loss_fn(output, target)+loss_VAE(mu=mu, logvar=logvar)
                        
                        #transformando para numpy para calcular as métricas
                        target = target.cpu().numpy().transpose(0, 2, 3, 1) # Convertendo para NHWC
                        predictions = output.cpu().numpy().transpose(0, 2, 3, 1)  # Convertendo para NHWC
                        for i in range(predictions.shape[0]):

                            if i == 1:
                                break
                            pred_img = predictions[i][::-1]
                            target_img = target[i][::-1]

                            # Normalizar se necessário (0-1)
                            if pred_img.max() > 1.0:
                                pred_img = pred_img / 255.0
                            if target_img.max() > 1.0:
                                target_img = target_img / 255.0
                            # Criar uma figura com 1 linha e 3 colunas
                            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

                            # Mostrar cada imagem em um subplot
                            axes[0].imshow(target_img)
                            axes[0].set_title("Target")
                            axes[0].axis('off')  # Desativar os eixos
                            axes[1].imshow(pred_img)
                            axes[1].set_title("Prediction*255")
                            axes[1].axis('off')
                            axes[2].imshow(pred_img*255)
                            axes[2].set_title("Prediction")
                            axes[2].axis('off')


                            # Ajustar o layout para evitar sobreposição
                            plt.tight_layout()

                            # Exibir o multiplot
                            plt.show()

                            #Salvar a imagem predita
                            #cv2.imwrite(f"{results_savedir}{model_name}_prediction_{batch_idx}_{i}.png", pred_img * 255)

                            # # Calcula a métrica
                            # psnr_value, ssim_value, uciqe_, uiqm = calculate_metrics(predictions, target)
                        
                            # psnr_list.append(psnr_value)
                            # ssim_list.append(ssim_value)
                            # uciqe_list.append(uciqe_)
                            # uiqm_list.append(uiqm)
                            
                else:
                        for batch_idx, (data, target) in tqdm(enumerate(test_loader_UIEB)):
                            if batch_idx == 1:
                                break
                            data, target = data.cuda(), target.cuda()
                            
                            output = model(data)
                            
                            loss = loss_fn(output, target)

                            #transformando para numpy para calcular as métricas
                            target = target.cpu().numpy().transpose(0, 2, 3, 1) # Convertendo para NHWC
                            predictions = output.cpu().numpy().transpose(0, 2, 3, 1)  # Convertendo para NHWC
                            for i in range(predictions.shape[0]):
                                if i == 1:
                                    break
                                pred_img = predictions[i][::-1]
                                target_img = target[i][::-1]

                                # Normalizar se necessário (0-1)
                                if pred_img.max() > 1.0:
                                    pred_img = pred_img / 255.0
                                if target_img.max() > 1.0:
                                    target_img = target_img / 255.0

                                # Criar uma figura com 1 linha e 3 colunas
                                fig, axes = plt.subplots(1, 3, figsize=(12, 4))

                                # Mostrar cada imagem em um subplot
                                axes[0].imshow(target_img)
                                axes[0].set_title("Target")
                                axes[0].axis('off')  # Desativar os eixos

                                axes[1].imshow(pred_img)
                                axes[1].set_title("Prediction*255")
                                axes[1].axis('off')

                                axes[2].imshow(pred_img*255)
                                axes[2].set_title("Prediction")
                                axes[2].axis('off')


                                # Ajustar o layout para evitar sobreposição
                                plt.tight_layout()

                                # Exibir o multiplot
                                plt.show()

                                #Salvar a imagem predita
                                #cv2.imwrite(f"{results_savedir}{model_name}_prediction_{batch_idx}_{i}.png", pred_img * 255)

                                # # Calcula a métrica
                                # psnr_value, ssim_value, uciqe_, uiqm = calculate_metrics(predictions, target)
                        
                                # psnr_list.append(psnr_value)
                                # ssim_list.append(ssim_value)
                                # uciqe_list.append(uciqe_)
                                # uiqm_list.append(uiqm)
                                            
    
       
                """
                #Ajustar script para apagar as variaveis sempre que salvar as metricas no disco  
                #Salvar a imagem predita
                cv2.imwrite(f"{results_savedir}{model_name}_prediction_{batch_idx}_{i}.png", pred_img * 255)
                
                # Calcula a métrica
                psnr_value, ssim_value, uciqe_, uiqm = calculate_metrics(predictions, target)
        
                psnr_list.append(psnr_value)
                ssim_list.append(ssim_value)
                uciqe_list.append(uciqe_)
                uiqm_list.append(uiqm)
    
    #Calcula a media de cada metrica
    avg_loss = sum(loss_list) / len(loss_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)
    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_uciqe = sum(uciqe_list) / len(uciqe_list)
    avg_uiqm = sum(uiqm_list) / len(uiqm_list)
    
    # Salvar métricas em um arquivo
    with open(f'{txt_savedir}{model_name}_metrics.txt', 'w') as f:"""
        # f.write(f"""avg_ssim:{avg_ssim}\navg_psnr:{avg_psnr}\navg_uciqe:{avg_uciqe}\navg_uiqm:{avg_uiqm}\navg_loss:{avg_loss}""")
        # print(f"Metrics for {model_name} saved to {txt_savedir}{model_name}_metrics.txt") 
    """ 
    avg_loss = 0.0
    avg_ssim = 0.0
    avg_psnr = 0.0
    avg_uciqe = 0.0
    avg_uiqm = 0.0
    
    psnr_list.remove(all)
    ssim_list.remove(all)
    uciqe_list.remove(all)
    uiqm_list.remove(all)
    """
    pass