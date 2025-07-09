# train.py
# cmd: python train.py --dataset cifar10 --snr_db 10 --compression_ratio 0.0417
# cmd: python train.py --dataset kodak --snr_db 10 --compression_ratio 0.0417
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from model import Autoencoder
from utils import *
from config import CONFIG
from data_load import load_cifar10_data, load_kodak_dataset
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def parse_args():
    """
    解析命令行参数，为训练提供配置。
    """
    parser = argparse.ArgumentParser(description="Train Deep JSCC model on different datasets.")
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'kodak'], 
                        help="Dataset to train on.")
    parser.add_argument('--compression_ratio', type=float, required=True, 
                        help="Bandwidth compression ratio (k/n). This is a required argument.")
    parser.add_argument('--snr_db', type=float, default=CONFIG['snr_db'], 
                        help="Signal-to-Noise Ratio in dB for training.")
    parser.add_argument('--channel_type', type=str, choices=['awgn', 'rayleigh'], default=CONFIG['channel_type'], 
                        help="Type of channel for training.")
    parser.add_argument('--epochs', type=int, default=CONFIG['epochs'], 
                        help="Number of training epochs.")
    parser.add_argument('--learning_rate', type=float, default=CONFIG['learning_rate'], 
                        help="Learning rate for the optimizer.")
    parser.add_argument('--batch_size', type=int, default=CONFIG['batch_size'], 
                        help="Batch size for training.")
    parser.add_argument('--norm_mode', type=str, default='dynamic', choices=['dynamic', 'paper'],
                        help="Normalization mode ('dynamic' or 'paper').")
    return parser.parse_args()

def train_model(epochs, batch_size, learning_rate, snr_db, channel_type, compression_ratio, dataset, normalization_mode):
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA (NVIDIA GPU) for training.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU) for training.")
    else:
        device = torch.device("cpu")
        print("Using CPU for training.")

    if dataset == 'cifar10':
        train_loader, test_loader = load_cifar10_data(batch_size)
    elif dataset == 'kodak':
        print("Warning: Using small Kodak dataset for both training and validation. This is for demonstration and will overfit.")
        kodak_loader = load_kodak_dataset(path='./kodak_dataset', batch_size=batch_size)
        train_loader, test_loader = kodak_loader, kodak_loader
    else:
        raise ValueError("Unsupported dataset specified.")

    # 从加载的数据中动态获取维度并计算k值
    sample_data, _ = next(iter(train_loader))
    _, C, H, W = sample_data.shape
    n_dim = C * H * W # 计算输入图像的总维度
    latent_H, latent_W = H // 4, W // 4 # 假设模型下采样4倍
    latent_channels_k = max(1, round(compression_ratio * n_dim / (latent_H * latent_W)))
    # 确保k值至少为1
    print(f"\n--- Training on {dataset.upper()} with: SNR={snr_db}dB, Channel={channel_type}, Target k/n={compression_ratio:.4f} (Model latent channels k={latent_channels_k}) ---")
    
    # 初始化模型
    
    model = Autoencoder(k=latent_channels_k, normalization_mode=normalization_mode).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    criterion = nn.MSELoss()

    train_losses, val_losses, val_psnrs, val_ssims = [], [], [], []
    best_loss = float('inf') # 初始化为正无穷大
    patience_counter = 0
    patience = 20 

    for epoch in range(epochs):
        model.train() # 训练模式
        total_train_loss = 0 # 初始化训练损失
        
        for data, target in train_loader: # 遍历训练数据
            optimizer.zero_grad() # 在计算新一轮梯度前，清空上一轮的梯度信息
            data, target = data.to(device), target.to(device) # 将这批次的输入数据和目标数据发送到我们选择的计算设备（GPU或CPU）
            output = model(data, snr_db=snr_db, channel_type=channel_type) # 前向传播。数据流过整个模型，得到重建图像，model.forward() 方法的简写
            loss = criterion(output, target) # 计算损失函数
            loss.backward() # 反向传播，计算梯度
            optimizer.step() # 更新模型参数
            total_train_loss += loss.item() # 累加当前批次的损失
        
        avg_train_loss = total_train_loss / len(train_loader) # 计算平均训练损失
        train_losses.append(avg_train_loss) # 输出当前epoch的平均训练损失

        # Validation
        model.eval() # 验证模式
        total_val_loss = 0 # 初始化验证损失
        psnr_list, ssim_list = [], []
        with torch.no_grad(): # 在验证阶段不需要计算梯度
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data, snr_db=snr_db, channel_type=channel_type)
                total_val_loss += criterion(output, target).item()
                for i in range(data.size(0)): # 遍历当前批次的每个图像
                    x = np.clip(data[i].cpu().numpy().transpose(1, 2, 0), 0, 1)
                    x_hat = np.clip(output[i].cpu().numpy().transpose(1, 2, 0), 0, 1)
                    psnr_list.append(compare_psnr(x, x_hat, data_range=1.0)) # 计算PSNR
                    ssim_list.append(compare_ssim(x, x_hat, data_range=1.0, channel_axis=-1, multichannel=True)) # 计算SSIM

        avg_val_loss = total_val_loss / len(test_loader) # 计算平均验证损失
        val_losses.append(avg_val_loss) # 输出当前epoch的平均验证损失
        """
        val_losses是一个在训练开始前创建的空列表。这行代码的作用是，将刚刚算出的本轮次的平均验证损失，
        作为一个新元素添加到val_losses列表的末尾。这样，每训练一轮，这个列表就会变长一点
        """
        avg_psnr = np.mean(psnr_list) # 计算平均PSNR
        val_psnrs.append(avg_psnr) # 输出当前epoch的平均PSNR
        avg_ssim = np.mean(ssim_list)
        val_ssims.append(avg_ssim)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val PSNR: {avg_psnr:.2f}dB | Val SSIM: {avg_ssim:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break
                
    return model, train_losses, val_losses, val_psnrs, val_ssims