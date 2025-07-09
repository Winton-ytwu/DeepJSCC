# plot_results.py
# cmd: python plot_result.py --dataset cifar10
import json
import os
import argparse
import re
from visualization import plot_final_comparison

def main():
    parser = argparse.ArgumentParser(description="Generate final comparison plot from saved results.")
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'kodak'],
                        help="The dataset for which to generate the plot ('cifar10' or 'kodak').")
    args = parser.parse_args()

    print(f"--- Generating Final Comparison Plot for {args.dataset.upper()} ---")
    results_dir = './results'
    
    # 1. 加载Benchmark结果，这部分不变
    
    benchmark_file = os.path.join(results_dir, f'benchmark_results_{args.dataset}.json')
    try:
        with open(benchmark_file, 'r') as f:
            benchmark_data = json.load(f)
        k_n_ratios = benchmark_data['k_n_ratios']
        results_jpeg = benchmark_data['jpeg']
        results_jpeg2000 = benchmark_data['jpeg2000']
        print(f"Benchmark results for {args.dataset.upper()} loaded successfully from {benchmark_file}")
    except FileNotFoundError:
        print(f"错误: 未找到基准测试文件 {benchmark_file}")
        print(f"请先运行 'python benchmark.py --dataset {args.dataset}'")
        return
    
    # 2. 智能扫描并加载所有已存在的Deep JSCC评估结果
     # 2. 扫描并区分两种模式的评估结果
    print("\nScanning for all evaluation results...")
    all_jscc_results_raw = {}
    
    # 匹配 ..._paper_... 或 ...（没有标签）... 的文件名
    pattern = re.compile(rf'evaluation_deep_jscc_{args.dataset}(_paper)?_snr(\d+)_awgn_kn([\d.]+)\.json')

    for filename in os.listdir(results_dir):
        match = pattern.match(filename)
        if match:
            mode, snr_str, ratio_str = match.groups()
            snr, ratio = int(snr_str), float(ratio_str)
            
            label = 'Deep JSCC (Paper P=1)' if mode else 'Deep JSCC (Dynamic P)'
            
            if label not in all_jscc_results_raw:
                all_jscc_results_raw[label] = {}
            if snr not in all_jscc_results_raw[label]:
                all_jscc_results_raw[label][snr] = {}

            eval_file_path = os.path.join(results_dir, filename)
            with open(eval_file_path, 'r') as f:
                eval_data = json.load(f)
            psnr_value = eval_data.get('awgn', {}).get('psnr', {}).get(str(snr))
            
            if psnr_value is not None:
                all_jscc_results_raw[label][snr][ratio] = psnr_value
    
     # 3. 整理数据
    all_jscc_results_for_plot = {}
    for label, snr_data in all_jscc_results_raw.items():
        all_jscc_results_for_plot[label] = {}
        for snr, ratio_data in snr_data.items():
            psnr_list = [ratio_data.get(r) for r in k_n_ratios]
            all_jscc_results_for_plot[label][str(snr)] = psnr_list
            
     # 4. 调用新的可视化函数绘图
    plot_final_comparison(
        k_n_ratios=k_n_ratios,
        all_jscc_results=all_jscc_results_for_plot,
        results_jpeg=results_jpeg,
        results_jpeg2000=results_jpeg2000,
        save_dir=results_dir,
        dataset=args.dataset
    )


    print(f"\n--- Final plot for {args.dataset.upper()} generated successfully! Check the 'results' folder. ---")


if __name__ == '__main__':
    main()