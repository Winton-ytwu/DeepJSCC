import torch
import torch.nn.functional as F
import numpy as np
import math
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


# AWGN Channel
def awgn_channel(z, snr_db):

    power = torch.mean(torch.abs(z) ** 2) # 计算实际平均功率P
    snr_linear = 10 ** (snr_db / 10) # 将dB转换为线性比例, P / σ²
    noise_power = power / snr_linear # σ²
    # torch.randn_like(real)生成一个与信号实部形状相同、符合标准正态分布（功率/方差=1）的“原始”噪声。
    # torch.sqrt(noise_power / 2) 计算出目标实部噪声的“标准差”。（总功率均分给实部和虚部，再开方得到标准差）
    # 两者相乘，将“原始”噪声缩放到期望的正确功率水平。
    noise_real = torch.randn_like(z.real) * torch.sqrt(noise_power / 2) # 最终噪声 = 标准噪声 * 目标标准差
    noise_imag = torch.randn_like(z.imag) * torch.sqrt(noise_power / 2)
    # 将独立的实部噪声和虚部噪声组合成一个复数噪声张量，用于添加到信号上
    noise = torch.complex(noise_real, noise_imag)

    return z + noise

# Rayleigh Fading Channel (slow fading)
def rayleigh_channel(z, snr_db):
    batch_size = z.shape[0] # 获取批次大小，为每一张图片成一个独立的衰落系数h
    power = torch.mean(torch.abs(z) ** 2) # 计算实际平均功率P
    snr_linear = 10 ** (snr_db / 10) # 将dB转换为线性比例, P / σ²
    noise_power = power / snr_linear # σ²
    
    # 生成复数衰落系数 h
    h_real = torch.randn(batch_size, 1, 1, 1, device=z.device) * (1 / math.sqrt(2)) # 实部的衰落系数
    h_imag = torch.randn(batch_size, 1, 1, 1, device=z.device) * (1 / math.sqrt(2)) # 虚部的衰落系数
    h = torch.complex(h_real, h_imag) # 将实部和虚部组合成复数衰落系数
    
    # 生成复数噪声
    noise_real = torch.randn_like(z.real) * torch.sqrt(noise_power / 2)
    noise_imag = torch.randn_like(z.imag) * torch.sqrt(noise_power / 2)
    noise = torch.complex(noise_real, noise_imag)
    
    return h * z + noise

def resize_input(x, target_size=(32, 32)):# 将输入图像调整为32x32
    return F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)


def psnr(img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0) -> float:
    """ 计算峰值信噪比 (PSNR)。
    Args:
        img1 (np.ndarray): 原始图像。
        img2 (np.ndarray): 重建图像。
        data_range (float): 图像数据范围，默认为1.0。
    Returns:
        float: 计算得到的PSNR值。
    """
    return compare_psnr(img1, img2, data_range=data_range)

def ssim(img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0, channel_axis: int = -1, multichannel: bool = True) -> float:
    # 将 multichannel 参数传递给底层的 compare_ssim 函数
    """ 计算结构相似性指数 (SSIM)。
    Args:
        img1 (np.ndarray): 原始图像。
        img2 (np.ndarray): 重建图像。
        data_range (float): 图像数据范围，默认为1.0。
        channel_axis (int): 通道轴，默认为-1。
        multichannel (bool): 是否为多通道图像，默认为True。
    Returns:
        float: 计算得到的SSIM值。
    """
    return compare_ssim(img1, img2, data_range=data_range, channel_axis=channel_axis, multichannel=multichannel)
