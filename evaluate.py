# evaluate.py

import torch
import numpy as np
import argparse
import os
import json
from tqdm import tqdm

from model import Autoencoder
from data_load import load_cifar10_data, load_kodak_dataset
from utils import psnr, ssim

def evaluate_model(model, test_loader, snr_list, channel_type, device, latent_channels_k):
    model.eval() # 设置模型为评估模式
    psnrs_over_snr = {snr: [] for snr in snr_list} # 存储每个SNR下的PSNR
    ssims_over_snr = {snr: [] for snr in snr_list} # 存储每个SNR下的SSIM
    
    with torch.no_grad(): # 在评估阶段不需要计算梯度
        for snr_db in tqdm(snr_list, desc=f"Evaluating on {channel_type} channel"): # 遍历每个SNR值
            for data, _ in test_loader: # 遍历每个SNR值下的测试数据
                data = data.to(device)
                output = model(data, snr_db=snr_db, channel_type=channel_type)
                for i in range(data.size(0)):
                    x = np.clip(data[i].cpu().numpy().transpose(1, 2, 0), 0, 1)
                    x_hat = np.clip(output[i].cpu().numpy().transpose(1, 2, 0), 0, 1)
                    psnrs_over_snr[snr_db].append(psnr(x, x_hat, data_range=1.0))
                    ssims_over_snr[snr_db].append(ssim(x, x_hat, data_range=1.0, channel_axis=-1, multichannel=True))
                    
    avg_psnrs = {str(snr): float(np.mean(psnrs_over_snr[snr])) for snr in snr_list} # 计算每个SNR下的平均PSNR
    avg_ssims = {str(snr): float(np.mean(ssims_over_snr[snr])) for snr in snr_list} # 计算每个SNR下的平均SSIM
    return avg_psnrs, avg_ssims

def main():
    parser = argparse.ArgumentParser(description="Evaluate Deep JSCC model and save results.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained .pth file.")
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'kodak'], help="Dataset the model was trained on.")
    parser.add_argument('--channel_types', nargs='+', default=['awgn', 'rayleigh'], help="List of channel types to evaluate.")
    parser.add_argument('--snr_min', type=int, default=0, help="Minimum SNR in dB for evaluation range.")
    parser.add_argument('--snr_max', type=int, default=20, help="Maximum SNR in dB for evaluation range.")
    parser.add_argument('--snr_step', type=int, default=1, help="Step size for SNR in dB.")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for evaluation.")
    parser.add_argument('--save_dir', type=str, default='./results', help="Directory to save evaluation results.")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA (NVIDIA GPU) for evaluation.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU) for evaluation.")
    else:
        device = torch.device("cpu")
        print("Using CPU for evaluation.")

    test_loader, H, W, C = (None, 0, 0, 0)
    if args.dataset == 'cifar10':
        _, test_loader = load_cifar10_data(args.batch_size)
        H, W, C = 32, 32, 3
        print("CIFAR-10 test data loaded.")
    elif args.dataset == 'kodak':
        test_loader = load_kodak_dataset(path='./kodak_dataset', batch_size=1)
        sample_data, _ = next(iter(test_loader)) # Get a sample to determine dimensions
        _, C, H, W = sample_data.shape
        print("Kodak test data loaded.")
    
    try:
        model_basename = os.path.basename(args.model_path) # 从包含文件夹的完整路径中，仅提取出文件名本身
        ratio_str = model_basename.split('_kn')[1].split('.pth')[0] # 提取k/n比率字符串
        compression_ratio = float(ratio_str) # 解析k/n比率
        
        # 根据模型结构动态计算k值
        latent_H, latent_W = H // 4, W // 4
        latent_channels_k = max(1, round(compression_ratio * (C * H * W) / (latent_H * latent_W)))
        print(f"Inferred k/n ratio: {compression_ratio}, using model parameter k={latent_channels_k}")
        
    except (IndexError, ValueError) as e:
        print(f"错误：无法从模型文件名 '{args.model_path}' 中解析出k/n比率。")
        exit()
        
    model = Autoencoder(k=latent_channels_k).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Model loaded successfully from {args.model_path}")

    snr_list = list(range(args.snr_min, args.snr_max + 1, args.snr_step))
    
    all_results = {}
    for channel_type in args.channel_types:
        avg_psnrs, avg_ssims = evaluate_model(model, test_loader, snr_list, channel_type, device, latent_channels_k)
        all_results[channel_type] = {'psnr': avg_psnrs, 'ssim': avg_ssims}

    model_name = os.path.splitext(os.path.basename(args.model_path))[0] # 从模型路径中提取文件名（去除扩展名），用于命名结果文件
    save_path = os.path.join(args.save_dir, f"evaluation_{model_name}.json") # 构建保存评估结果的完整路径
    os.makedirs(os.path.dirname(save_path), exist_ok=True) # 如果保存目录不存在则自动创建
    with open(save_path, 'w') as f: # 打开结果文件，准备写入
        json.dump(all_results, f, indent=4) # 将评估结果以JSON格式写入文件
    print(f"\nEvaluation results for {model_name} saved to {save_path}") # 打印保存成功的信息


if __name__ == "__main__":
    main()