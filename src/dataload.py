import os
import random
import math
import cv2
import numpy as np
import glob
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.utils.data as data

def load_image_paths(dataset_path, dataset="all",task="train",split=False):
    """
    dataset_path: endereço do dataset raiz
    dataset: "all", "UIEB", "RUIE", "SUIM"
    task: "train", "val"

    """
    image_paths = []
    if dataset == "all":
        # Constrói os padrões de caminho para os arquivos .jpg e .png dentro das pastas train e train/images
        pattern1_jpg = os.path.join(dataset_path, "*", f"{task}", "*.jpg")
        pattern2_jpg = os.path.join(dataset_path, "*", f"{task}", "images", "*.jpg")
        pattern1_png = os.path.join(dataset_path, "*", f"{task}", "*.png")
        pattern2_png = os.path.join(dataset_path, "*", f"{task}", "images", "*.png")
        pattern3_jpg = os.path.join(dataset_path, "*", "*",f"{task}", "*.jpg")

        
        # Encontra todos os arquivos .jpg e .png correspondentes aos padrões
        image_paths.extend(glob.glob(pattern1_jpg))
        image_paths.extend(glob.glob(pattern2_jpg))
        image_paths.extend(glob.glob(pattern1_png))
        image_paths.extend(glob.glob(pattern2_png))
        image_paths.extend(glob.glob(pattern3_jpg))
    elif dataset == "SUIM":
         pattern2_jpg = os.path.join(dataset_path, "*", f"{task}", "images", "*.jpg")
         image_paths.extend(glob.glob(pattern2_jpg))
    elif dataset == "UIEB":
        pattern1_png = os.path.join(dataset_path, "*", f"{task}", "*.png")
        image_paths.extend(glob.glob(pattern1_png))
    elif dataset == "RUIE":
        pattern3_jpg = os.path.join(dataset_path, "*", "*",f"{task}", "*.jpg")
        image_paths.extend(glob.glob(pattern3_jpg))
    
    # Embaralha os caminhos das imagens
    random.shuffle(image_paths)
    if split == True:
        # Divide os dados em 80% para treino e 20% para teste
        split_index = int(len(image_paths) * 0.8)
        train_paths = image_paths[:split_index]
        test_paths = image_paths[split_index:]
        
        return train_paths, test_paths
    else:
        return image_paths
    

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




class load_data(data.Dataset):
    def __init__(self, input_data_low, input_data_high):
        self.input_data_low = input_data_low
        self.input_data_high = input_data_high
        print("Total training examples:", len(self.input_data_high))
        self.transform=A.Compose(
            [
                A.Resize (height=256, width=256),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                ToTensorV2(),
            ]
        )
        


    def __len__(self):  
        return len(self.input_data_low)
    
    def light_adjusts(self,image):
        
        mean = np.round(np.mean(image)/255,1)
        std = np.round(np.std(image)/255,2)
        self.transform_light_high = A.Compose([
            A.ColorJitter(brightness=high_light_adjust(mean,std), contrast=0, saturation=0, hue=0, p=1.0),
        ])

        self.transform_light_low = A.Compose([
            A.ColorJitter(brightness=low_light_adjust(mean,std), contrast=0, saturation=0, hue=0, p=1.0),
        ])
        data_low = self.transform_light_low(image=image)["image"]
        data_high = self.transform_light_high(image=image)["image"]

        return  data_low, data_high

    def __getitem__(self, idx):
        seed = torch.random.seed()
        data_low = cv2.imread(self.input_data_low[idx])

        data_low=data_low[:,:,::-1].copy()
        data_low, data_high = self.light_adjusts(image=data_low)
        random.seed(1)
        
        data_low = self.transform(image=data_low)["image"]/255
  

        return [data_low, data_high,data_color,data_blur]



class load_data_test(data.Dataset):
    def __init__(self, input_data_low, input_data_high):
        self.input_data_low = input_data_low
        self.input_data_high = input_data_high
        print("Total test-training examples:", len(self.input_data_high))
        self.transform=A.Compose(
            [
                A.Resize (height=256, width=256),
                ToTensorV2(),
            ]
        )


    def __len__(self):
        return len(self.input_data_low)
    
    def light_adjusts(self,image):
        
        mean = np.round(np.mean(image)/255,1)
        std = np.round(np.std(image)/255,2)
        self.transform_light_high = A.Compose([
            A.ColorJitter(brightness=high_light_adjust(mean,std), contrast=0, saturation=0, hue=0, p=1.0),
        ])

        self.transform_light_low = A.Compose([
            A.ColorJitter(brightness=low_light_adjust(mean,std), contrast=0, saturation=0, hue=0, p=1.0),
        ])
        data_low = self.transform_light_low(image=image)["image"]
        data_high = self.transform_light_high(image=image)["image"]

        return  data_low, data_high


    def __getitem__(self, idx):
        seed = torch.random.seed()
        data_low = cv2.imread(self.input_data_low[idx])

        data_low=data_low[:,:,::-1].copy()
        #data_low, data_high = self.light_adjusts(image=data_low)
        _, data_high = self.light_adjusts(image=data_low)
        random.seed(1)
        
        data_low = self.transform(image=data_low)["image"]/255
        
       

        return [data_low, data_high,data_color,data_blur, self.input_data_low[idx]]
