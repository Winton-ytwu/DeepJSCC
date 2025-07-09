# visualization.py (最终工具库版)

import matplotlib.pyplot as plt
import numpy as np
import nbformat as nbf
import os

def create_notebook_with_plots(train_losses, val_losses, val_psnrs, val_ssims, save_dir, snr_db, channel_type, compression_ratio):
    """
    这个函数用于绘制和保存“训练过程”的指标（Loss, PSNR vs. Epochs），
    并生成一个Jupyter Notebook。
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建一个空的 Jupyter Notebook
    nb = nbf.v4.new_notebook()

    # 添加标题和参数信息，在Jupyter Notebook报告的开头写好标题和信息
    title_text = f"""
# Training Metrics Plot

* **SNR**: {snr_db} dB
* **Channel Type**: {channel_type}
* **Compression Ratio**: {compression_ratio:.4f} (k/n)
"""
    
    nb.cells.append(nbf.v4.new_markdown_cell(title_text))# 将这个Markdown文本作为一个单元格添加到Notebook中

    # 从train.py传递过来的、包含所有性能指标的Python列表（如train_losses, val_psnrs等），转换成一个包含Python代码的长字符串
    data_injection_code = f"""
# Data from training run
train_losses = {train_losses}
val_losses = {val_losses}
val_psnrs = {val_psnrs}
val_ssims = {val_ssims}
save_dir = '{save_dir}'
"""
    nb.cells.append(nbf.v4.new_code_cell(data_injection_code))

    # 添加绘图代码块
    plotting_code = """
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 6))

# 绘制训练损失和验证损失曲线
plt.subplot(1, 3, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.title('Loss vs. Epochs')

# 绘制验证PSNR曲线
plt.subplot(1, 3, 2)
plt.plot(val_psnrs, label="Validation PSNR", color='green')
plt.xlabel('Epochs')
plt.ylabel('PSNR (dB)')
plt.legend()
plt.grid(True)
plt.title('Validation PSNR vs. Epochs')

# 绘制验证SSIM曲线
plt.subplot(1, 3, 3)
plt.plot(val_ssims, label="Validation SSIM", color='red')
plt.xlabel('Epochs')
plt.ylabel('SSIM')
plt.legend()
plt.grid(True)
plt.title('Validation SSIM vs. Epochs')

plt.tight_layout()
save_path = f"{save_dir}/training_metrics_plot.png"
plt.savefig(save_path)
print(f"Training metrics plot saved to {save_path}")
plt.show()
plt.close()
"""
    nb.cells.append(nbf.v4.new_code_cell(plotting_code))

    # 将生成的 notebook 保存为 .ipynb 文件
    notebook_filename = os.path.join(save_dir, "training_metrics_notebook.ipynb")
    with open(notebook_filename, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Training metrics notebook saved as {notebook_filename}")


def plot_final_comparison(k_n_ratios, all_jscc_results, results_jpeg, results_jpeg2000, save_dir, dataset):
    """
    这个新函数专门用于绘制最终的、包含多组实验对比的性能图。
    """
    plt.figure(figsize=(10, 8))
    
    # 定义颜色和标记的映射，以保持图表的一致性
    snr_colors = {'0': 'green', '10': 'orange', '20': 'blue'}
    # 不同模式用不同标记来区分
    mode_markers = {'Deep JSCC (Dynamic P)': 's', 'Deep JSCC (Paper P=1)': 'o'} 
    
    # --- 绘制所有Deep JSCC模型的曲线 ---
    # 遍历我们整理好的、包含所有JSCC实验结果的大字典
    for name, results_dict in all_jscc_results.items():
        marker = mode_markers.get(name, 'X') # 获取该模式对应的标记，如果找不到则用'X'
        # 遍历该模式下不同SNR的结果
        for snr_str, psnr_list in results_dict.items():
            # 过滤掉值为None的数据点，以便绘制不完整的曲线
            valid_points = [(r, p) for r, p in zip(k_n_ratios, psnr_list) if p is not None]
            if valid_points:
                ratios, psnrs = zip(*valid_points)
                # 为每个SNR使用固定的颜色
                color = snr_colors.get(snr_str, 'black')
                plt.plot(ratios, psnrs, marker=marker, linestyle='-', color=color, label=f'{name} (SNR={snr_str}dB)')
        
    # --- 绘制JPEG和JPEG2000基准的曲线 ---
    for snr_str, psnr_list in results_jpeg.items():
        color = snr_colors.get(snr_str, 'black')
        # 使用与论文一致的五角星'p'标记
        plt.plot(k_n_ratios, psnr_list, marker='p', linestyle='--', color=color, label=f'JPEG (SNR={snr_str}dB)')

    for snr_str, psnr_list in results_jpeg2000.items():
        color = snr_colors.get(snr_str, 'black')
        # 使用菱形'd'标记
        plt.plot(k_n_ratios, psnr_list, marker='d', linestyle='--', color=color, label=f'JPEG2000 (SNR={snr_str}dB)')

    # 设置图表的各种标签和格式
    plt.xlabel("Bandwidth Compression Ratio (k/n)")
    plt.ylabel("PSNR (dB)")
    plt.title(f"Performance Comparison on {dataset.upper()} Dataset")
    plt.legend() # 显示图例
    plt.grid(True) # 添加网格线
    plt.ylim(10, 55) # 设置Y轴范围
    
    # 根据数据集动态命名并保存最终的图片文件
    save_path = os.path.join(save_dir, f"final_comparison_{dataset}.png")
    plt.savefig(save_path)
    plt.close()
    
    print(f"\nFinal comparison plot saved as {save_path}")
def plot_robustness_curves(results_list, save_dir, k_n_ratio):
    """
    这个新函数专门用于绘制Figure 4这样的“鲁棒性”曲线。
    """
    plt.figure(figsize=(10, 8))
    
    for result in results_list:
        # 从文件名中解析出训练时的SNR
        model_name = os.path.splitext(os.path.basename(result['path']))[0]
        try:
            # 智能地处理带或不带模式标签的文件名
            snr_train_str = model_name.split('_snr')[1].split('_')[0]
            snr_train = int(snr_train_str)
        except (IndexError, ValueError):
            snr_train = "Unknown"
        
        # 提取awgn信道下的psnr数据
        awgn_psnr_data = result['data'].get('awgn', {}).get('psnr', {})
        if awgn_psnr_data:
            # 将json的键（字符串）转换为整数用于绘图，并排序
            snr_test_list = sorted([int(k) for k in awgn_psnr_data.keys()])
            psnr_list = [awgn_psnr_data[str(k)] for k in snr_test_list]
            plt.plot(snr_test_list, psnr_list, marker='o', linestyle='-', label=f'Deep JSCC (SNR_train={snr_train}dB)')

    plt.xlabel("Test SNR (dB)")
    plt.ylabel("PSNR (dB)")
    plt.title(f"Robustness to SNR Mismatch (k/n = {k_n_ratio:.4f})")
    plt.legend()
    plt.grid(True)
    
    # 动态命名保存的图片文件
    save_path = os.path.join(save_dir, f"robustness_plot_kn_{k_n_ratio:.4f}.png")
    plt.savefig(save_path)
    plt.close()
    
    print(f"Robustness plot saved as {save_path}")