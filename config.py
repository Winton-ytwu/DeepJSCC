# config.py (最终版)

CONFIG = {
    'channel_type': 'awgn',  # 默认信道类型: 'awgn' 或 'rayleigh'
    'batch_size': 64,        # 默认批处理大小
    'learning_rate': 1e-3,   # 默认学习率
    'epochs': 600,           # 默认训练回合数 (您可以根据需要增加)
    'snr_db': 10,            # 默认信噪比 (dB)
}