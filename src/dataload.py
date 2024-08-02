import os
import random
import math
import cv2
import numpy as np
import glob
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.utils.data as data
import Albumentations as A
from torch.utils.data import DataLoader

def load_image_paths(dataset_path="data/", dataset: str= None, task="train",split=False):

    """
    Load paired datasets for training and testing.
        dataset_path: endereço do dataset raiz
        dataset: "UIEB", "EUVP", "HICRD", "LSUI", "TURBID"
        task: "train", "val"
        split:  True, False #split in 80% train and 20% test
    return:
        list[paired_data]: data_raw, data_ref
    """

    image_paths_raw,image_paths_ref = [],[]
    # Constrói os padrões de caminho para os arquivos .jpg e .png dentro das pastas train e train/images
    """if dataset == "EUVP":
        pattern_png_raw = os.path.join(dataset_path, "*","/Paired/", "*","/train_A/", "*.jpg")
        image_paths_raw.extend(glob.glob(pattern_png_raw))
        pattern_png_ref = os.path.join(dataset_path, "*","/Paired/", "*","/train_B/", "*.jpg")
        image_paths_ref.extend(glob.glob(pattern_png_ref))"""
    if dataset == "UIEB":
        pattern_png_raw = os.path.join(dataset_path, "*", "/raw-890/", "*.png")
        image_paths_raw.extend(glob.glob(pattern_png_raw))
        pattern_png_ref = os.path.join(dataset_path, "*", "/reference-890/", "*.png")
        image_paths_ref.extend(glob.glob(pattern_png_ref))
        """elif dataset == "HICRD":
        pattern_png_raw = os.path.join(dataset_path, "*", "/trainA_paired/", "*.png")
        image_paths_raw.extend(glob.glob(pattern_png_raw))
        pattern_png_ref = os.path.join(dataset_path, "*", "/trainB_paired/", "*.png")
        image_paths_ref.extend(glob.glob(pattern_png_ref))"""
    
    elif dataset == "TURBID":
        pattern_png_raw = os.path.join(dataset_path, "*", "*.jpg")
        image_paths_raw.extend(glob.glob(pattern_png_raw))
        image_paths_raw = [path for path in image_paths_raw if not path.endswith('ref.jpg')]
        pattern_png_ref = os.path.join(dataset_path, "*", "ref.jpg")
        for i in range(len(image_paths_raw)):
            image_paths_ref.append(image_paths_ref[0])
    elif dataset == "LSUI":
        pattern_png_raw = os.path.join(dataset_path, "*", "/input/", "*.jpg")
        image_paths_raw.extend(glob.glob(pattern_png_raw))
        pattern_png_ref = os.path.join(dataset_path, "*", "/GT/", "*.jpg")
        image_paths_ref.extend(glob.glob(pattern_png_ref))
    else:
        raise ValueError("Invalid dataset name\nPlease choose from ['UIEB', 'EUVP', 'HICRD', 'LSUI', 'TURBID']\n or add your own dataset")
    
    # Embaralha os caminhos das imagens
    if split == True:
        # Divide os dados em 80% para treino e 20% para teste
        split_index = int(len(image_paths_raw) * 0.8)
        train_paths = image_paths_raw[:split_index]
        test_paths = image_paths_ref[split_index:]
        train_paths = image_paths_raw[:split_index]
        test_paths = image_paths_ref[split_index:]
        
        return train_paths, test_paths
    else:
        return image_paths_raw, image_paths_ref
    

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


"""Creating dataloaders"""


#Dataloader for LSUI
def create_dataloader(dataset_name: str =None, dataset_path: str =None, batch_size=16, num_workers=4):
   #load paths
    train_paths, test_paths = load_image_paths(dataset_path=dataset_path, dataset = dataset_name, task=True, split=True)
    
    #initialize it
    train_dataset = load_data(input_data_low=train_paths[0], input_data_high=train_paths[1])
    test_dataset = load_data(input_data_low=test_paths[0], input_data_high=test_paths[1])

    #creating the DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

#dataloader UIEB
train_loader_UIEB, test_loader_UIEB = create_dataloader(dataset_name="Uieb", dataset_path="data/UIEB")

#dataloader TURBID
train_loader_TURBID, test_loader_TURBID = create_dataloader(dataset_name="Turbid", dataset_path="data/TURBID")

#dataloader LSUI
train_loader_LSUI, test_loader_LSUI = create_dataloader(dataset_name="Lsui", dataset_path="data/LSUI")






