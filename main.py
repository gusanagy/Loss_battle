import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from loss import *
from models import *
from src.dataload import *
from metrics.metrics import *
from tqdm import tqdm


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"Rank {rank} initialized.")

def cleanup():
    dist.destroy_process_group()

def train_models(rank, world_size, epochs, loss_fn = None, model_name=None, model=None,dataset_name="UIEB", dataset_path="data"):
    setup(rank, world_size)
    ckpt_savedir='output/ckpt_battle/'
    if not os.path.exists(ckpt_savedir):
        os.makedirs(ckpt_savedir)
    results_savedir='output/results_battle/'
    if not os.path.exists(results_savedir):
        os.makedirs(results_savedir)

    modelos = load_models()
    loss_battle = []

    loss_battle.extend(build_perceptual_losses(rank= rank))
    loss_battle.extend(build_channel_losses(rank = rank))
    loss_battle.extend(build_structural_losses(rank = rank))

    print(f"{len(loss_battle)} loss functions to train with")
    print(f"{len(modelos)} models to train with")
    
    #dataloader UIEB
    train_loader_UIEB, test_loader_UIEB, sampler = create_dataloader(dataset_name=dataset_name, dataset_path=dataset_path,world_size=world_size,rank=rank,ddp=True)
    for model in modelos:
        ddp_model = DDP(model.cuda(rank), device_ids=[rank])
        optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=0.001, weight_decay=1e-5)
        for loss_fn in loss_battle:
            print(f"Training {model.__class__.__name__} with {loss_fn.__class__.__name__}")
            model_name = model.__class__.__name__ + "_" + loss_fn.__class__.__name__
            
            for epoch in range(epochs):
                ddp_model.train()
                sampler.set_epoch(epoch)

                if rank == 0:
                    epoch_pbar = tqdm(total=len(train_loader_UIEB), desc=f"Epochs for {model_name}", position=0)
                    
                for batch_idx, (data, target) in enumerate(train_loader_UIEB):
                    data, target = data.cuda(rank), target.cuda(rank)
                    optimizer.zero_grad()

                    output = ddp_model(data)
                    loss = loss_fn(output, target).requires_grad_(True)
                    
                    loss.backward()
                    optimizer.step()

                    if batch_idx % 100 == 0:
                        print(f"Rank {rank}, Epoch [{epoch}/{epochs}], Batch [{batch_idx}/{len(train_loader_UIEB)}], Loss: {loss.item()}")
                    if rank == 0:
                        epoch_pbar.update(1)
            
            if rank == 0:
                epoch_pbar.close()
            ##Salve Dir para salvar os checkpoints
            if rank == 0:
                print(f"Testando o modelo {model_name}")
                # Salvar o estado do modelo original, não o DDP
                torch.save(model.state_dict(), f"{ckpt_savedir}{model_name}_ckpt.pth")
                psnr_list, ssim_list, uciqe_list, uiqm_list = [], [], [], []
                # Avaliar o modelo
                model.eval()
                with torch.no_grad():
                    for batch_idx, (data, target) in tqdm(enumerate(test_loader_UIEB)):
                        data, target = data.cuda(rank), target.cpu().numpy()
                        predictions = model(data).cpu().numpy()
                        #calcula a metrica
                        
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

            cleanup()

# def train(rank, world_size, epochs):
    
#     modelos = load_models()
#     loss_battle = []

#     loss_battle.extend(build_perceptual_losses())
#     loss_battle.extend(build_channel_losses())
#     loss_battle.extend(build_structural_losses())
#     print(f"{len(loss_battle)} loss functions to train with")
#     print(f"{len(modelos)} models to train with")
#     for model in modelos:
#         for loss_fn in loss_battle:
#             print(f"Training {model.__class__.__name__} with {loss_fn.__class__.__name__}")
#             model_name = model.__class__.__name__ + "_" + loss_fn.__class__.__name__
#             train_one_model(rank, world_size, epochs, loss_fn, model_name, model)        
        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs per model")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--gpus", type=int, default=2, help="Number of GPUs per node")
    args = parser.parse_args()

    world_size = args.nodes * args.gpus

    torch.multiprocessing.spawn(train_models, args=(world_size, args.epochs), nprocs=args.gpus, join=True)



