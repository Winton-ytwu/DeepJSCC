# plot_figure4.py

import json
import os
import argparse
import re
from visualization import plot_robustness_curves 

def main():
    parser = argparse.ArgumentParser(description="Generate robustness plot (like Figure 4).")
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'kodak'])
    parser.add_argument('--compression_ratio', type=float, required=True, 
                        help="The k/n ratio for which to generate the plot.")
    args = parser.parse_args()

    results_dir = './results'
    print(f"--- Generating Robustness Plot for k/n = {args.compression_ratio:.4f} ---")

    results_to_plot = []
    
    pattern_str = rf'evaluation_deep_jscc_{args.dataset}(_paper|_dynamic)?_snr(\d+)_awgn_kn{args.compression_ratio:.4f}\.json$'
    pattern = re.compile(pattern_str)

    for filename in os.listdir(results_dir):
        if pattern.match(filename):
            eval_file_path = os.path.join(results_dir, filename)
            try:
                with open(eval_file_path, 'r') as f:
                    eval_data = json.load(f)
                results_to_plot.append({'path': eval_file_path, 'data': eval_data})
                print(f"  Loaded evaluation file: {filename}")
            except Exception as e:
                print(f"Warning: Could not load or parse file {eval_file_path}. Error: {e}")

    if not results_to_plot:
        print(f"Error: No evaluation files found for the specified dataset and k/n ratio: {args.compression_ratio}")
        return
        
    # 调用新的绘图函数，并传入k/n比率用于标题
    plot_robustness_curves(results_to_plot, results_dir, args.compression_ratio)

if __name__ == '__main__':
    main()