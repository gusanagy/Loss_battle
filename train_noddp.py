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

def train_models(epochs: int=100, loss_fn=None, model_name=None, model=None, dataset_name="UIEB", dataset_path="data"):
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
    #loss_battle.extend(build_channel_losses(rank = device))
    #loss_battle.extend(build_structural_losses(rank = device))

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
                for batch_idx, (data, target) in enumerate(train_loader_UIEB):
                    data, target = data.cuda(), target.cuda()
                    optimizer.zero_grad()
                    output = model(data)
                    
                    loss = loss_fn(output, target).requires_grad_(True)

                    loss.backward()
                    optimizer.step()
                    ##Optionally print loss information
                    # if batch_idx % 100 == 0:
                    #     print(f"Epoch [{epoch}/{epochs}], Batch [{batch_idx}/{len(train_loader_UIEB)}], Loss: {loss.item()}")


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
