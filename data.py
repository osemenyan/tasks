from pathlib import Path
from tkinter import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as v2
torch.manual_seed(123)
from PIL import Image

train_path = Path('./base/custom_image_dataset/train')
val_path = Path('./base/custom_image_dataset/train')
def remove(img):
    img_array = np.array(img)
    height, width = img_array.shape[:2]
   
    img_array[height-16:, width-16:] = img_array.mean()
    return Image.fromarray(img_array)


    
def create_dataloaders():
    train_tf = v2.Compose([
    v2.RandomResizedCrop(size=(40, 40), antialias=True),
    v2.Lambda(remove), 
    v2.RandomHorizontalFlip(p=0.5),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.ToTensor(),
    v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
              ]) # TODO: 定义训练集的数据预处理与增强
    val_tf = v2.Compose([
    v2.RandomResizedCrop(size=(40, 40), antialias=True),
    v2.FiveCrop(24),                    
    v2.Lambda(lambda crops: crops[0]), 
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
          ]) # TODO: 定义验证集的数据预处理

    train_dataset = ImageFolder(root=str(Path('./base/custom_image_dataset/train')), transform= train_tf) # TODO: 加载训练集，并确保应用训练集的 transform
    val_dataset = ImageFolder(root=str(Path('./base/custom_image_dataset/train')), transform= val_tf) # TODO: 加载验证集

    train_loader = DataLoader(train_dataset,shuffle=True,batch_size=32) # TODO: 创建训练集 dataloader
    val_loader = DataLoader(val_dataset,shuffle=True,batch_size=32) # TODO: 创建验证集 dataloader

    return train_loader, val_loader




