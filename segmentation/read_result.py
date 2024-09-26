import os
import pandas as pd
import numpy as np

def get_val_acc_for_epoch(directory_path, epoch):
    epoch = int(epoch) # turn to int to remove 0 in case 07, 08, etc
    val_log_path = os.path.join(directory_path, 'val.log')
    with open(val_log_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            if parts[0] == str(epoch):
                return parts[1]
    return None


def print_aligned(peak_mem, mean_mem, std_mem):
    print(f"\n{'Peak mem':<20}{'Mean and deviation':<30}")
    mean_std_str = f"{mean_mem:.4f} ± {std_mem:.4f}"  
    print(f"{peak_mem:<20}{mean_std_str:<30}")


def extract_config(directory_path):
    config_path = os.path.join(directory_path, 'config.yaml')
    with open(config_path, 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if parts[0] == 'seed_everything:':
                seed = parts[1]
            elif parts[0] == 'num_of_finetune:':
                num_of_finetune = parts[1]
            elif parts[0] == 'backbone:':
                model = parts[1]
            elif parts[0] == 'name:':
                dataset = parts[1]
            elif parts[0] == 'with_SVD_with_var_compression:':
                with_SVD_with_var_compression = parts[1]
            elif parts[0] == 'with_HOSVD_with_var_compression:':
                with_HOSVD_with_var_compression = parts[1]
            elif parts[0] == 'with_grad_filter:':
                with_grad_filter = parts[1]
            elif parts[0] == 'SVD_var:':
                SVD_var = parts[1]
            elif parts[0] == 'filt_radius:':
                filt_radius = parts[1]
            elif parts[0] == 'exp_name:':
                exp_name = parts[1]
            
        if with_SVD_with_var_compression == 'false' and with_HOSVD_with_var_compression == 'false' and with_grad_filter == 'false':
            base = 'true'
        else: base = 'false'
    return seed, num_of_finetune, model, dataset, with_SVD_with_var_compression, with_HOSVD_with_var_compression, with_grad_filter, base, SVD_var, filt_radius, exp_name

def calculate_mean_and_deviation(data):
    mean = np.mean(data)
    squared_diff = sum((x - mean) ** 2 for x in data)
    variance = squared_diff / (len(data) - 1)
    deviation = np.sqrt(variance)
    # Round the results to 2 decimal places
    # mean = round(mean, 2)
    # deviation = round(deviation, 2)

    return mean, deviation

def get_mem(directory_path, unit):
    """
    unit: 'Byte', 'KB', or 'MB'
    """
    mem_log_path = os.path.join(directory_path, f'mem_log/activation_memory_{unit}.log')
    with open(mem_log_path, 'r') as file:
        lines = file.readlines()
        lines = lines[1:]
        mem=[]
        for line in lines:
            parts = line.strip().split()
            mem.append(float(parts[1]))
    mean, deviation = calculate_mean_and_deviation(mem)
    return max(mem), sum(mem), mean, deviation

import shutil

def clear_mem_log(root_directory):
    """
    Remove all the `mem_log` folders within experimental folders
    """
    def process_directory(current_directory):
        for entry in sorted(os.listdir(current_directory)):
            entry_path = os.path.join(current_directory, entry)
            if 'mem_log' in entry:
                if os.path.exists(os.path.join(current_directory, 'mem_log')):
                    shutil.rmtree(os.path.join(current_directory, 'mem_log'))
            elif os.path.isdir(entry_path):
                process_directory(entry_path)

    process_directory(root_directory)
    print("All mem_log folder are removed")

def get_results(root_directory, unit):
    results = []
    def process_directory(current_directory):
        for entry in sorted(os.listdir(current_directory)):
            entry_path = os.path.join(current_directory, entry)
            if 'mem_log' in entry:
                if os.path.exists(os.path.join(current_directory, 'mem_log/delete')):
                    shutil.rmtree(os.path.join(current_directory, 'mem_log/delete'))
                peak_mem, total_mem, mean_mem, std_mem = get_mem(current_directory, unit)
                print(f"==============Experiment Name: {current_directory}==============")

                divisor = 1
                peak_mem /= divisor
                mean_mem /= divisor
                std_mem /= divisor
                total_mem /= divisor
                print_aligned(peak_mem, mean_mem, std_mem)

                results.append({
                    'Experiment Name:': current_directory,
                    'peak_mem': peak_mem,
                    'mean_mem': mean_mem,
                    'std_mem': std_mem,
                    'mean_mem and std_mem': f"{mean_mem:.2f} ± {std_mem:.2f}",
                })
                
                print("====================================================================================================")
            elif os.path.isdir(entry_path):
                process_directory(entry_path)

    process_directory(root_directory)

    # Export results to Excel
    df = pd.DataFrame(results)
    df.to_excel('results.xlsx', index=False)
    print("Results have been exported to results.xlsx")


root_directory = "runs"
# root_directory = "runs/hosvd_10L_deeplabv3_r18-d8_512x512_20k_voc12aug"
get_results(root_directory, unit='MB')
clear_mem_log(root_directory)



