# data_load.py
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision import transforms
from PIL import Image
import os

# --- 为CIFAR10创建一个自定义的包装器，以适应自编码器任务 ---
class AutoencoderCIFAR10(CIFAR10):
    def __getitem__(self, index):
        img, _ = super().__getitem__(index) # 调用父类方法获取(图像,标签)，并忽略标签
        return img, img # 返回图像本身作为输入和目标

def load_cifar10_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor() # ToTensor会自动完成归一化和维度重排(HWC -> CHW)
    ])
    train_dataset = AutoencoderCIFAR10(root='./data', train=True, download=True, transform=transform)
    # CIFAR-10测试集使用自编码器包装器
    test_dataset = AutoencoderCIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Kodak数据集的加载函数保持不变
class KodakDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)
        return image_tensor, image_tensor

def load_kodak_dataset(path, batch_size=1):
    transform = transforms.Compose([transforms.ToTensor()])
    kodak_dataset = KodakDataset(root_dir=path, transform=transform)
    data_loader = DataLoader(kodak_dataset, batch_size=batch_size, shuffle=False)
    return data_loader