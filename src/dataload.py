import os
import random
import math
import cv2
import numpy as np
import glob
from torch.utils.data import DataLoader
import torch.utils.data as data
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

# import Albumentations as A
from torch.utils.data import DataLoader

def load_image_paths(dataset_path="data", dataset: str = None, task="train", split=False):
    """
    Load paired datasets for training and testing.
    
    Args:
        dataset_path (str): Endereço do dataset raiz.
        dataset (str): "UIEB", "EUVP", "HICRD", "LSUI", "TURBID".
        task (str): "train", "val".
        split (bool): True para dividir em 80% treino e 20% teste.
    
    Returns:
        list[paired_data]: data_raw, data_ref
    """

    image_paths_raw, image_paths_ref = [], []
    
    if dataset == "UIEB":
        pattern_png_raw = os.path.join(dataset_path, dataset, "raw-890", "*.png")
        image_paths_raw.extend(glob.glob(pattern_png_raw))
        
        pattern_png_ref = os.path.join(dataset_path, dataset, "reference-890", "*.png")
        image_paths_ref.extend(glob.glob(pattern_png_ref))
    
    elif dataset == "TURBID":
        pattern_jpg_raw = os.path.join(dataset_path, dataset, "*", "*.jpg")
        all_image_paths_raw = glob.glob(pattern_jpg_raw)
        image_paths_raw = [path for path in all_image_paths_raw if not path.endswith('ref.jpg')]
        
        pattern_jpg_ref = os.path.join(dataset_path, dataset, "*", "ref.jpg")
        image_paths_ref = glob.glob(pattern_jpg_ref)
    
    elif dataset == "LSUI":
        pattern_jpg_raw = os.path.join(dataset_path, dataset, "input", "*.jpg")
        image_paths_raw.extend(glob.glob(pattern_jpg_raw))
        
        pattern_jpg_ref = os.path.join(dataset_path, dataset, "GT", "*.jpg")
        image_paths_ref.extend(glob.glob(pattern_jpg_ref))
    
    else:
        raise ValueError("Invalid dataset name\nPlease choose from ['UIEB', 'EUVP', 'HICRD', 'LSUI', 'TURBID']\n or add your own dataset")
    
    if split:
        split_index = int(len(image_paths_raw) * 0.8)
        train_paths_raw = image_paths_raw[:split_index]
        test_paths_raw = image_paths_raw[split_index:]
        
        train_paths_ref = image_paths_ref[:split_index]
        test_paths_ref = image_paths_ref[split_index:]
        
        return train_paths_raw, train_paths_ref, test_paths_raw, test_paths_ref
    else:
        return image_paths_raw, image_paths_ref

def save_image_path_file(image_paths, filename):
    """
    Salva uma lista de endereços de imagens em um arquivo .txt.

    Args:
        image_paths (list): Lista de endereços de imagens.
        filename (str): Nome do arquivo .txt onde a lista será salva.
    """
    with open(filename, 'w') as file:
        for path in image_paths:
            file.write(f"{path}\n")
def load_image_paths_file(filename):
    """
    Carrega uma lista de endereços de imagens a partir de um arquivo .txt.

    Args:
        filename (str): Nome do arquivo .txt de onde a lista será carregada.

    Returns:
        list: Lista de endereços de imagens.
    """
    with open(filename, 'r') as file:
        image_paths = [line.strip() for line in file.readlines()]
    return image_paths
def file_exists(filename):
    """
    Verifica se um arquivo existe.

    Args:
        filename (str): Nome do arquivo a ser verificado.

    Returns:
        bool: True se o arquivo existe, False caso contrário.
    """
    return os.path.isfile(filename)
def check_splits(dataset_path="data", dataset_name: str = None):
    if file_exists(f"data/{dataset_name}/{dataset_name}_train_raw.txt") or file_exists(f"data/{dataset_name}/{dataset_name}_train_ref.txt") or file_exists(f"data/{dataset_name}/{dataset_name}_test_raw.txt") or file_exists(f"data/{dataset_name}/{dataset_name}_test_ref.txt"):
        train_raw = load_image_paths_file(f"data/{dataset_name}/{dataset_name}_train_raw.txt")
        train_ref = load_image_paths_file(f"data/{dataset_name}/{dataset_name}_train_ref.txt")
        test_raw = load_image_paths_file(f"data/{dataset_name}/{dataset_name}_test_raw.txt")
        test_ref = load_image_paths_file(f"data/{dataset_name}/{dataset_name}_test_ref.txt")
    else:
        train_raw, train_ref, test_raw, test_ref = load_image_paths(dataset_path=dataset_path, dataset=dataset_name, split=True)
        save_image_path_file(train_raw, f"data/{dataset_name}/{dataset_name}_train_raw.txt")
        save_image_path_file(train_ref, f"data/{dataset_name}/{dataset_name}_train_ref.txt")
        save_image_path_file(test_raw, f"data/{dataset_name}/{dataset_name}_test_raw.txt")
        save_image_path_file(test_ref, f"data/{dataset_name}/{dataset_name}_test_ref.txt")
    return train_raw, train_ref, test_raw, test_ref

def list_images(directory):
    """
    Lists all images in the given directory with extensions .png or .jpg.

    :param directory: The directory to search for images.
    :return: A list of file paths to the images.
    """
    image_extensions = ('.png', '.jpg', '.jpeg')
    image_paths = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_paths.append(os.path.join(root, file))

    return image_paths


class PairedImageDataset(Dataset):
    def __init__(self, raw_paths, ref_paths, transform=None):
        self.raw_paths = raw_paths
        self.ref_paths = ref_paths
        self.transform =  transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.raw_paths)

    def __getitem__(self, idx):
        raw_image = Image.open(self.raw_paths[idx]).convert("RGB")
        ref_image = Image.open(self.ref_paths[idx]).convert("RGB")

        if self.transform:
            raw_image = self.transform(raw_image)
            ref_image = self.transform(ref_image)

        return raw_image, ref_image

# Função de transformação

"""Creating dataloaders"""


#Dataloader for LSUI
def create_dataloader(dataset_name: str =None, dataset_path: str =None, batch_size=16, num_workers=4, ddp:bool = True, world_size=None, rank=None):
   
   #load paths
    #train_raw, train_val,test_raw, test_val= load_image_paths(dataset_path=dataset_path, dataset = dataset_name, split=True)
    train_raw, train_val, test_raw, test_val = check_splits(dataset_path=dataset_path, dataset_name = dataset_name)
    print(f"Train raw: {len(train_raw)} Train ref: {len(train_val)} \nTest raw: {len(test_raw)} Test ref: {len(test_val)}\n")
    #initialize it
    train_dataset = PairedImageDataset(raw_paths=train_raw, ref_paths=train_val)
    test_dataset = PairedImageDataset(raw_paths=test_raw, ref_paths=test_val)

    #creating the DataLoaders
    if ddp:
        #sampler
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(train_dataset,sampler=sampler, batch_size=batch_size, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if ddp:
        return train_loader, test_loader, sampler
    else:
        return train_loader, test_loader





