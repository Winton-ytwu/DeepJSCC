# benchmark.py

# 模拟一个理想化的传统数字通信系统（JPEG/JPEG2000 + 理想信道编码），
# 为训练的Deep JSCC模型提供一个性能参照物或“跑分基准”。
# cmd: python benchmark.py --dataset cifar10
import numpy as np
import json
import os
import subprocess  # 用于调用外部命令行工具，如opj_compress
import tempfile    # 用于创建临时文件（已弃用，但保留导入以备查）
from tqdm import tqdm  # 用于显示美观的进度条
from PIL import Image  # Pillow库，用于基础的图像打开和保存操作
import glymur        # 用于处理JPEG2000格式文件的库
import io            # 用于在内存中读写数据，类似一个内存中的文件
import uuid          # 用于生成唯一的ID，以创建不会冲突的临时文件名
import argparse      # 用于创建灵活的命令行接口，如此处的 --dataset 参数
import multiprocessing
from data_load import load_cifar10_data, load_kodak_dataset
from utils import psnr

def get_jpeg_psnr(original_image_np, target_bpp):
    """
    接收一张原始图像和一个目标码率(bpp)，通过二分搜索找到满足该码率预算的
    最佳JPEG压缩质量, 并返回其对应的PSNR值。
    """
    
    # 将输入的、像素值在[0,1]范围的浮点数Numpy数组，转换为[0,255]范围的8位整数数组
    original_image_uint8 = (np.clip(original_image_np, 0, 1) * 255).astype(np.uint8)
    # 将Numpy数组转换为Pillow图像对象，以便使用其强大的图像处理功能
    pil_image = Image.fromarray(original_image_uint8)

    # 如果目标码率小于等于0，意味着没有预算，直接返回一个代表失败的低PSNR值
    '''论文第5页描述的失败处理逻辑:“...we assume that the image cannot be reliably transmitted 
    and each color channel is reconstructed to the mean value of all the pixels for that channel.”'''
    # target_bpp = R * 3
    if target_bpp <= 0: # “target bits per pixel” 目标每像素比特数
        mean_color = np.mean(original_image_np, axis=(0, 1))
        # original_image_np: 这是输入的原始图像，一个形状为(高度, 宽度, 3)的NumPy数组。
        # np.mean(...): NumPy的求平均值函数
        # axis=(0, 1): 指定在高度和宽度维度上求平均，得到一个形状为(3,)的数组，表示每个颜色通道的平均值。
        return psnr(original_image_np, np.full_like(original_image_np, mean_color), data_range=1.0)
    
    # 计算将目标码率（比特/像素）转换为整个文件的目标大小（字节）
    # original_image_np.shape 返回一个元组，包含图像的高度、宽度和通道数
    # height 变量会得到第一个值（例如 512）
    # width 变量会得到第二个值（例如 768）
    # _ (下划线) 是一个Python中的常用惯例，用作一个“一次性的”或“我不在乎的”变量
    height, width, _ = original_image_np.shape # 从输入的NumPy图像数组中，获取其高度和宽度
    '''
    --original_image_np 是一个代表图像的NumPy数组，其形状（shape）通常是一个包含三个值的元组（tuple），
    即 (高度, 宽度, 通道数)。例如，一张Kodak图片可能是 (512, 768, 3)
    --height, width, _ = ... 是一种Python的元组解包 (Tuple Unpacking) 技术。
    它会把.shape返回的元组中的值，依次赋给左边的变量。
    '''
    num_pixels = height * width # 计算图像的总像素数（高度 * 宽度）
    target_bytes = (target_bpp * num_pixels) / 8 # 将目标码率转换为字节数（比特/像素 * 总像素数 / 8）
    '''
    --target_bpp: 根据信道容量算出的每个像素最多可以使用的比特数。
    --target_bpp * num_pixels: 计算出整张图片最多可以包含的总比特数。
    --/ 8: 因为1字节（Byte）等于8比特（bit），所以除以8，将总比特数转换为总字节数。
    --target_bytes就是我们压缩后的JPEG文件不能超过的最大尺寸
    '''

    # 初始化二分搜索的参数
    low_q, high_q = 1, 95  # JPEG质量参数的有效范围，初始化二分搜索的范围
    best_q = 0             #  初始化一个变量，用于存储我们找到的满足预算的最佳质量值
    """
    初始化为0（一个无效的质量值）。在后续的二分搜索中，一旦我们找到了一个可行的q值
    （即压缩后的文件大小 ≤ target_bytes），我们就会更新best_q。如果搜索结束后，
    best_q仍然是0，就意味着连最低质量q=1都无法满足预算，即压缩失败
    """
    best_reconstructed_image_np = None # 用于记录最佳质量对应的重建图像

    # 开始二分搜索循环
    while low_q <= high_q:
        # 取当前搜索范围的中间值作为本次尝试的质量参数。
        # // 是整数除法，确保q是整数。
        q = (low_q + high_q) // 2 # 取中间的质量值进行尝试
        
        if q == 0: break # 安全检查，防止q变成0，避免无限循环
        
        # 创建一个内存中的二进制“文件”，用于临时存储压缩后的图像数据，避免频繁读写硬盘
        buffer = io.BytesIO()
        # 使用Pillow库，以质量q来保存JPEG图片到内存中
        pil_image.save(buffer, format='jpeg', quality=q)
        
        # # buffer.tell() 会返回当前内存“文件”的大小（字节数）。
        # 检查这个大小是否超出了我们的预算。
        if buffer.tell() <= target_bytes:
            # 如果符合预算，这是一个可行的方案。记录下并尝试寻找更高的质量
            # 1. 更新最佳质量值o
            best_q = q 
            # 2. 将内存文件的指针移回开头，以便读取
            buffer.seek(0)
            # 3. 打开这个内存中的JPEG数据，将其解码变回图像，并存为Numpy数组。
            best_reconstructed_image_np = np.array(Image.open(buffer), dtype=np.float32) / 255.0
            # 4. 既然质量q可行，就尝试寻找更高的质量。
            #    将搜索范围的下限提高到q+1，相当于“扔掉”了所有比q低的质量区域。
            low_q = q + 1 # 提高质量下限，在右半边继续搜索
        else:
            # 如果文件太大，超预算了，说明质量q太高了。
            # 我们需要降低质量。我们将搜索范围的上限降低到q-1，
            # 相当于“扔掉”了所有比q高的质量区域。
            high_q = q - 1 
    
    # 如果循环结束后best_q仍然是0，说明预算太小，连最低质量(q=1)的JPEG都存不下
    if best_q == 0:
        # 返回代表“压缩失败”的PSNR值
        mean_color = np.mean(original_image_np, axis=(0, 1))
        return psnr(original_image_np, np.full_like(original_image_np, mean_color), data_range=1.0)
    """
    -- np.full_like会创建一个与原始图像original_image_np形状完全相同的数组，
       并用我们上一步计算出的mean_color来填充所有像素。最终结果就是一张没有任何细节的、均匀的纯色色块图
    -- psnr函数会计算原始图像和这张纯色图之间的PSNR值。
       由于纯色图没有任何细节，所以PSNR值会非常低，
       这表示压缩失败，无法可靠地重建原始图像。
       这个恒定的低PSNR值，就代表了该条件下基准测试的“性能下限”或“失败状态”。
       这符合论文中描述的逻辑：“...we assume that the image cannot be reliably transmitted 
       and each color channel is reconstructed to the mean value of all the pixels for that channel.”
       也就是当预算太小，无法存储任何有效信息时，
       我们就假设图像无法可靠传输，并将每个颜色通道的像素值都设置为该通道的平均值。
    """
    
    # 否则，返回我们找到的最佳质量方案所对应的PSNR值
    return psnr(original_image_np, best_reconstructed_image_np, data_range=1.0)
    

def get_jpeg2000_psnr(original_image_np, target_bpp):
    """
    接收一张原始图像和一个目标码率(bpp)，通过调用外部opj_compress工具
    来执行JPEG2000压缩，并返回其对应的PSNR值。
    """
    original_image_uint8 = (np.clip(original_image_np, 0, 1) * 255).astype(np.uint8)
    pil_image = Image.fromarray(original_image_uint8)
    """
        --np.clip(...): 确保输入的浮点数在[0, 1]范围内。
        --* 255: 将[0, 1]范围的浮点数放大到[0, 255]范围。
        --.astype(np.uint8): 将浮点数转换为8位无符号整数, 这是标准图像的像素格式。
        --Image.fromarray(...): 使用Pillow库, 从这个整数NumPy数组创建一个图像对象, 以便我们后续可以将其保存为.bmp等格式。
    """
    
    # 原始未压缩图像的码率是24 bpp (每个像素3个通道，每个通道8比特)
    raw_bpp = 24.0
    
    # 根据目标码率处理边界情况
    # 根据目标码率target_bpp，计算出需要传递给opj_compress工具的文件大小压缩比
    if target_bpp >= raw_bpp:
        # 如果预算比原始图像还大，执行无损压缩（压缩比为1）
        compression_ratio = 1.0
    elif target_bpp <= 0:
        # 如果没有预算，直接返回失败值
        mean_color = np.mean(original_image_np, axis=(0, 1))
        return psnr(original_image_np, np.full_like(original_image_np, mean_color), data_range=1.0)
    else:
        # 正常情况下，计算opj_compress需要的文件大小压缩比
        compression_ratio = raw_bpp / target_bpp

    # 创建唯一的临时文件名，避免多进程冲突
    temp_id = uuid.uuid4() # 生成一个基于随机数的、几乎不可能重复的通用唯一标识符（UUID）
    temp_input_filename = f"{temp_id}_in.bmp"
    temp_output_filename = f"{temp_id}_out.jp2"
    
    reconstructed_image_np = None
    
    # 使用try...finally确保临时文件在操作结束后一定会被删除
    try:
        # 1. 将原始图像数据保存为一个无损的BMP临时文件
        pil_image.save(temp_input_filename, format='BMP')
        
        # 2. 构建将要执行的命令行指令
        cmd = ['opj_compress', '-i', temp_input_filename, '-o', temp_output_filename, '-r', str(compression_ratio)]
        # 构建一个列表，其中包含了要在命令行中执行的命令和所有参数

        # 3. 使用subprocess.run调用外部程序，并隐藏其输出信息
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        """
        --check=True: 如果opj_compress执行失败（例如，因为压缩比过高而报错），这个参数会让subprocess.run抛出一个异常，然后程序会跳转到except块进行失败处理。
        --stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL: 将opj_compress的正常输出和错误输出都“丢弃”，保持我们自己程序的终端输出干净。
        """
        # 4. 如果压缩成功，用glymur读取生成的jp2文件
        jp2_data = glymur.Jp2k(temp_output_filename)
        reconstructed_image_np = jp2_data[:] / 255.0
        # 从glymur对象中提取出解码后的像素数据，并将其归一化到[0, 1]范围。
    
    except (Exception, subprocess.CalledProcessError):
        # 如果任何一步（特别是subprocess）出错，则进入失败逻辑，生成一个纯色的“失败图像”
        mean_color = np.mean(original_image_np, axis=(0, 1))
        reconstructed_image_np = np.full_like(original_image_np, mean_color)
    
    finally:
        # 5. 无论成功与否，都删除临时文件
        if os.path.exists(temp_input_filename):
            os.remove(temp_input_filename)
        if os.path.exists(temp_output_filename):
            os.remove(temp_output_filename)

    # 计算PSNR值
    calculated_psnr = psnr(original_image_np, reconstructed_image_np, data_range=1.0)
    
    # 为PSNR设置一个合理的上限（80dB），避免因近无损压缩导致的数值溢出
    return min(calculated_psnr, 80.0)


def run_theoretical_benchmark(k_n_ratio, snr_db, test_loader, num_workers):# 
    """
    这是连接通信理论和图像压缩的桥梁函数。
    """
    # 将dB单位的信噪比转换为线性的功率比值
    snr_linear = 10**(snr_db / 10.0)
    # 应用香农信道容量公式，计算每次信道使用最多能传多少比特
    capacity = np.log2(1 + snr_linear)
    # 将理论值转换为具体的码率预算(比特/像素)
    max_bpp = k_n_ratio * capacity * 3
    
    # --- 并行处理的核心修改 ---
    
    # 1. 准备所有待处理的任务（每张图片都是一个任务）
    tasks = [] # 用于存储所有待处理的任务列表
    print("Preparing tasks for parallel processing...")
    for data, _ in test_loader: # 遍历数据加载器，一次取出一批（batch）数据
        for i in range(data.size(0)): # 遍历这一批数据中的每一张图片
            image_np = data[i].cpu().numpy().transpose(1, 2, 0) # 将PyTorch张量格式的图片，转换为NumPy数组格式
            tasks.append((image_np, max_bpp)) # 将图片和码率预算打包成一个任务
            #（图片数据，码率预算）作为一个元组添加到tasks列表中。如果测试集有10,000张图片，这个列表最终就会包含10,000个任务
            
    # 2. 创建一个进程池，并将任务分发给所有进程并行处理
    """
    创建一个“进程池”，您可以把它想象成雇佣了num_workers个（例如8个）员工。
    with语句能确保在计算完成后，这些员工会被安全地“解散”，不会占用系统资源
    """
    print(f"Executing {len(tasks)} tasks using {num_workers} parallel workers...")
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(process_single_image, tasks), total=len(tasks), desc=f"Benchmarking k/n={k_n_ratio:.4f}, SNR={snr_db}dB"))
        """
        向进程池分发任务的命令。imap是一种高效的分发方式，它会像发牌一样，把tasks列表中的每一个任务（(图片, 预算)）逐一地、不间断地分发给空闲的员工（进程）。
        每个员工在拿到任务后，都会执行我们在脚本顶层定义的process_single_image这个“工作指南”函数
        """
    # 3. 收集所有进程返回的结果并计算平均值
    all_psnrs_jpeg = [res[0] for res in results]
    # 使用列表推导式，从results列表中，将每个结果元组的第一个元素（JPEG的PSNR）提取出来，形成一个新的列表
    all_psnrs_jpeg2000 = [res[1] for res in results]
    # 提取每个结果元组的第二个元素（JPEG2000的PSNR）形成另一个列表

    return np.mean(all_psnrs_jpeg), np.mean(all_psnrs_jpeg2000)
    # 函数使用NumPy的mean函数，分别计算这两个列表的算术平均值，并将其作为这个实验数据点的最终性能指标返回

# 每个并行进程要执行的“工人”函数
def process_single_image(args):# 接收一个包含图像数据和最大码率预算的元组
    image_np, max_bpp = args # 解包这个元组，得到单张图片的NumPy数组和最大码率预算
    psnr_jpeg = get_jpeg_psnr(image_np, max_bpp) # 调用之前定义的函数，计算JPEG压缩的PSNR值
    psnr_jpeg2000 = get_jpeg2000_psnr(image_np, max_bpp) # 调用之前定义的函数，计算JPEG2000压缩的PSNR值
    return psnr_jpeg, psnr_jpeg2000


# 当这个脚本作为主程序运行时，执行以下代码
# Python的标准写法，确保只有当这个文件被直接运行时，内部的代码才会被执行。
# 如果它被其他文件import，这部分代码则不会运行
if __name__ == '__main__':
    # 使用 argparse 创建命令行接口，让我们可以用 --dataset 和 --workers 参数
    parser = argparse.ArgumentParser(description="Run theoretical benchmarks for JPEG/JPEG2000.") # 创建一个命令行参数解析器对象
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'kodak'],
                        help="Dataset to use for the benchmark ('cifar10' or 'kodak').")
    """
    --dataset: 添加一个名为--dataset的参数，它的值必须是字符串（type=str），可选值为'cifar10'或'kodak'（choices=...），默认值为'cifar10'。
    """
    parser.add_argument('--workers', type=int, default=multiprocessing.cpu_count(), 
                        help="Number of parallel workers to use. Defaults to all available CPU cores.")
    """
    --workers: 添加一个名为--workers的参数，它的值必须是整数（type=int），默认值为当前系统的CPU核心数（multiprocessing.cpu_count()）。
    """
    args = parser.parse_args() # 解析命令行参数，返回一个包含所有参数的对象，可以通过args.dataset和args.workers来访问这些值

    # 根据传入的参数，调用对应的数据加载函数
    test_loader = None
    if args.dataset == 'cifar10':
        print("--- Loading CIFAR-10 Dataset ---")
        _, test_loader = load_cifar10_data(batch_size=256)
    elif args.dataset == 'kodak':
        print("--- Loading Kodak Dataset ---")
        if not os.path.isdir('./kodak_dataset'):
            print("错误: 未找到 './kodak_dataset' 文件夹。请先下载Kodak数据集。")
            exit()
        test_loader = load_kodak_dataset(path='./kodak_dataset', batch_size=1)
    
    # 设置好要测试的所有snr和k/n的值
    snr_values_db = [0, 10, 20]
    k_n_ratios = [0.0417, 0.0833, 0.1667, 0.25, 0.3333, 0.4167, 0.5]
    
    # 准备用于存储结果的字典
    results_jpeg, results_jpeg2000 = {}, {}

    print(f"--- Starting Benchmark Calculation on {args.dataset.upper()} Dataset ---")
    
    # 通过嵌套的for循环，遍历所有实验条件
    for snr in snr_values_db:
        psnr_j_list, psnr_j2k_list = [], []
        for ratio in k_n_ratios:
            # 调用主函数，执行一次完整的基准测试计算
            avg_psnr_jpeg, avg_psnr_jpeg2000 = run_theoretical_benchmark(ratio, snr, test_loader, num_workers=args.workers)
            psnr_j_list.append(avg_psnr_jpeg) # 将单次实验返回的平均PSNR值存入临时列表
            psnr_j2k_list.append(avg_psnr_jpeg2000) # 将单次实验返回的平均PSNR值存入临时列表
            # 打印单次实验的结果
            print(f"Result: k/n={ratio:.4f}, SNR={snr}dB -> JPEG PSNR={avg_psnr_jpeg:.2f}dB, JPEG2000 PSNR={avg_psnr_jpeg2000:.2f}dB")
        
        results_jpeg[str(snr)] = psnr_j_list
        results_jpeg2000[str(snr)] = psnr_j2k_list
        # 当一个SNR下的所有k/n比率都测试完毕后，将包含7个PSNR值的临时列表，存入最终的结果字典中。
        # 键是SNR值（转换为字符串以便JSON兼容），值是PSNR列表
    # 将所有计算结果汇总到一个字典中
    all_benchmark_results = {
        'k_n_ratios': k_n_ratios,
        'jpeg': results_jpeg,
        'jpeg2000': results_jpeg2000
    }
    
    # 根据数据集动态命名输出文件，并用json.dump将结果保存到文件中
    save_path = f'./results/benchmark_results_{args.dataset}.json'
    os.makedirs(os.path.dirname(save_path), exist_ok=True) # 确保保存目录存在，如果不存在则创建
    with open(save_path, 'w') as f:
        json.dump(all_benchmark_results, f, indent=4) # 将结果以JSON格式写入文件，缩进为4个空格，便于阅读
    
    print(f"\n--- {args.dataset.upper()} benchmark results saved to {save_path} ---")