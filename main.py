# main.py
# cmd: python main.py --dataset cifar10 --snr_db 10 --compression_ratio 0.0417 --norm_mode paper
import torch
import os
# 确保从我们最终版的train.py中导入parse_args
from train import train_model, parse_args
from visualization import create_notebook_with_plots

def main():
    # 1. 解析所有命令行参数，包括 --norm_mode
    args = parse_args()
    
    # 2. 将所有参数传递给训练函数
    model, train_losses, val_losses, val_psnrs, val_ssims = train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        snr_db=args.snr_db,
        channel_type=args.channel_type,
        compression_ratio=args.compression_ratio,
        dataset=args.dataset,
        normalization_mode=args.norm_mode
    )
    
    # 3. 创建保存结果的文件夹
    results_dir = './results'
    models_dir = './models'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # --- 关键修正：根据模式智能地构建文件名 ---
    # 只有在进行'paper'模式的A/B对比实验时，才在文件名中添加特殊标签
    if args.norm_mode == 'paper':
        mode_tag = f'_{args.norm_mode}' # mode_tag 变量的值是 '_paper'
    else:
        # 对于默认的'dynamic'模式，不添加任何标签，以保持与您旧文件的兼容性
        mode_tag = '' # mode_tag 变量的值是一个空字符串

    # 在f-string中统一使用 mode_tag 变量
    model_filename = f"deep_jscc_{args.dataset}{mode_tag}_snr{int(args.snr_db)}_{args.channel_type}_kn{args.compression_ratio:.4f}.pth"
    model_save_path = os.path.join(models_dir, model_filename)
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel saved to {model_save_path}")

    # 4. 可视化报告的文件夹名也随之更新
    report_save_dir = os.path.join(results_dir, f"training_report_{os.path.splitext(model_filename)[0]}")
    create_notebook_with_plots(
        train_losses=train_losses,
        val_losses=val_losses,
        val_psnrs=val_psnrs,
        val_ssims=val_ssims,
        save_dir=report_save_dir,
        snr_db=args.snr_db,
        channel_type=args.channel_type,
        compression_ratio=args.compression_ratio
    )
    print("\n--- Training script finished. ---")


if __name__ == "__main__":
    main()