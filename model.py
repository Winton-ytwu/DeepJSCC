# model.py 定义神经网络结构

import torch
import torch.nn as nn
from utils import awgn_channel, rayleigh_channel # 只导入需要的信道函数

class Autoencoder(nn.Module):
    # 在构造函数中增加一个参数来选择模式，默认为我们之前的'dynamic'方法
    def __init__(self, k, normalization_mode='dynamic'):
        super(Autoencoder, self).__init__()
        self.k = k # 让模型实例“记住”自己的k值
        self.normalization_mode = normalization_mode # 保存归一化模式

        # Encoder: 移除所有BatchNorm和MaxPool，使用带步长的卷积进行下采样
        self.encoder = nn.Sequential(
            # 第1层: 输入(3通道, 32x32) -> 输出(16通道, 16x16)。stride=2实现尺寸减半
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.PReLU(),
            # 第2层: 输入(16通道, 16x16) -> 输出(32通道, 8x8)。stride=2再次实现尺寸减半
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.PReLU(),
            # 第3层: 输入(32通道, 8x8) -> 输出(32通道, 8x8)。stride=1，尺寸不变
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
            # 第4层: 输入(32通道, 8x8) -> 输出(32通道, 8x8)。stride=1，尺寸不变
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
            # 第5层(输出层): 输出(2*k通道, 8x8)。这是编码器的最终输出，包含了k个复数的信息
            nn.Conv2d(32, 2 * self.k, kernel_size=5, stride=1, padding=2),
            nn.PReLU()
        )

        # Decoder: 与Encoder镜像对称
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d 是卷积的逆操作，用于上采样（放大尺寸）
            # 第1层: 输入(2*k通道, 8x8) -> 输出(32通道, 8x8)。stride=1，尺寸不变
            nn.ConvTranspose2d(2 * self.k, 32, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),

            # 第2层: 输入(32通道, 8x8) -> 输出(32通道, 8x8)。stride=1，尺寸不变
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),

            # 第3层: 新增的、与Encoder镜像对称的层, 输入(32通道, 8x8) -> 输出(32通道, 8x8)
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
            
            # 第4层: 输入(32通道, 8x8) -> 输出(16通道, 16x16)。stride=2实现尺寸加倍
            # output_padding=1 用于精确控制输出尺寸，确保与编码器对应层匹配
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.PReLU(),
            
            # 第5层: 输入(16通道, 16x16) -> 输出(3通道, 32x32)。stride=2再次实现尺寸加倍
            nn.ConvTranspose2d(16, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
            
            # 最后的Sigmoid激活函数，将输出像素值约束在[0, 1]范围内，与归一化后的输入图像对应
            nn.Sigmoid()
        )

    def forward(self, x, snr_db=10, channel_type="awgn"):
        # 1. 编码器输出一个交错的实数张量，形状为 (batch, 2*k, H_latent, W_latent)
        encoded_interleaved = self.encoder(x)
        
        # 2. "打包"：将交错的实数张量转换为标准的复数张量z
        z_tilde = torch.complex(encoded_interleaved[:, 0::2], encoded_interleaved[:, 1::2])
        
        # 根据模式选择不同的归一化方法
        z = None
        if self.normalization_mode == 'paper':
            # 强制归一化
            k_total = self.k * (x.shape[2] // 4) * (x.shape[3] // 4)
            power_constraint_p = 1.0 # 设定P=1
            
            norm_factor = torch.sqrt(torch.sum(torch.abs(z_tilde)**2, dim=[1,2,3], keepdim=True))
            epsilon = 1e-8 # 避免分母为0
            z = torch.sqrt(torch.tensor(k_total * power_constraint_p, device=z_tilde.device)) * z_tilde / (norm_factor + epsilon)
        else: # 默认使用 'dynamic' 模式
            z = z_tilde

        # 3. 将复数张量z送入选择的信道模型
        z_noisy = None
        if channel_type == "awgn":
            z_noisy = awgn_channel(z, snr_db)
        elif channel_type == "rayleigh":
            z_noisy = rayleigh_channel(z, snr_db)
        else:
            raise ValueError("Unsupported channel type.")
            
        # 4. "解包"：将加噪后的复数张量z_noisy，转换回解码器需要的交错实数张量格式
        x_noisy = torch.empty_like(encoded_interleaved)
        x_noisy[:, 0::2] = torch.real(z_noisy)
        x_noisy[:, 1::2] = torch.imag(z_noisy)
        
        # 5.  解码器接收加噪后的张量，重建图像
        decoded = self.decoder(x_noisy)
        return decoded