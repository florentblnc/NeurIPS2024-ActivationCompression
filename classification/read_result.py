import os
import re
import pandas as pd
import numpy as np

def extract_epoch_from_filename(directory_path):
    pattern = re.compile(r'epoch=(\d+)-val-acc=\d+\.\d+\.ckpt')
    for filename in os.listdir(os.path.join(directory_path, 'checkpoints')):
        match = pattern.match(filename)
        if match:
            return match.group(1)
    return None

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


def print_aligned(epoch_value, val_acc, peak_mem, mean_mem, std_mem, total_mem):
    print(f"\n{'Epoch':<10}{'Acc':<20}{'Peak mem':<20}{'Mean and deviation':<30}{'Total mem':<12}")
    mean_std_str = f"{mean_mem:.4f} ± {std_mem:.4f}"
    print(f"{epoch_value:<10}{val_acc:<20}{peak_mem:<20}{mean_std_str:<30}{total_mem:<12}")


def extract_config(directory_path):
    config_path = os.path.join(directory_path, 'config.yaml')
    # Initialize variables
    seed = None
    num_of_finetune = None
    model = None
    dataset = None
    with_SVD = None
    with_HOSVD = None
    with_grad_filter = None
    base = None
    explained_variance_threshold = None
    filt_radius = None
    exp_name = None
    max_epochs = None
    
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
            elif parts[0] == 'with_SVD:':
                with_SVD = parts[1]
            elif parts[0] == 'with_HOSVD:':
                with_HOSVD = parts[1]
            elif parts[0] == 'with_grad_filter:':
                with_grad_filter = parts[1]
            elif parts[0] == 'explained_variance_threshold:':
                explained_variance_threshold = parts[1]
            elif parts[0] == 'filt_radius:':
                filt_radius = parts[1]
            elif parts[0] == 'exp_name:':
                exp_name = parts[1]
            elif parts[0] == 'max_epochs:':
                max_epochs = parts[1]
                
            
        if with_SVD == 'false' and with_HOSVD == 'false' and with_grad_filter == 'false':
            base = 'true'
        else: base = 'false'
    return seed, num_of_finetune, model, dataset, with_SVD, with_HOSVD, with_grad_filter, base, explained_variance_threshold, filt_radius, exp_name, max_epochs

def calculate_mean_and_deviation(data):
    """
    Calculate mean and standard deviation of the given data.

    Parameters:
        data (list): A list of numeric values.

    Returns:
        tuple: A tuple containing the mean and standard deviation of the data.

    Example:
        If data = [1, 2, 3, 4, 5], the calculated mean and standard deviation will be:
        (3.0, 1.58)
    """
    # Calculate the mean
    mean = np.mean(data)

    squared_diff = sum((x - mean) ** 2 for x in data)
    # Calculate the variance
    variance = squared_diff / (len(data) - 1)

    # Calculate the standard deviation
    deviation = np.sqrt(variance)

    # Round the results to 2 decimal places
    # mean = round(mean, 2)
    # deviation = round(deviation, 2)

    return mean, deviation

def get_mem(directory_path, unit, is_decomposition=True):
    """ unit: 'Byte', 'KB', or 'MB' """
    mem_log_path = os.path.join(directory_path, f'activation_memory_{unit}.log')
    if is_decomposition:
        with open(mem_log_path, 'r') as file:
            lines = file.readlines()
            mem=[]
            for line in lines:
                parts = line.strip().split()
                mem.append(float(parts[1]))
        mean, deviation = calculate_mean_and_deviation(mem)
        return max(mem), sum(mem), mean, deviation
    else:
        with open(mem_log_path, 'r') as file:
            for line in file:
                match = re.search(r'Activation memory is ([\d.]+) MB', line)
                if match: return float(match.group(1))  # Trả về số Activation memory dưới dạng float


def read_result(root_directory, unit):
    results = []
    def process_directory(current_directory):
        # Iterate through all items in the current directory
        for entry in sorted(os.listdir(current_directory)):
            entry_path = os.path.join(current_directory, entry)
            if entry == 'val.log':
                # If 'val.log' is encountered, take necessary actions:
                seed, num_of_finetune, model, dataset, with_SVD, \
                with_HOSVD, with_grad_filter, base, explained_variance_threshold, \
                    filt_radius, exp_name, max_epochs = extract_config(current_directory)

                print(f"==============Experiment Name: {exp_name}==============")
                epoch_value = extract_epoch_from_filename(current_directory)
                val_acc = get_val_acc_for_epoch(current_directory, epoch_value)

                if epoch_value:
                    print(f"{'Seed:':<20}{seed:<20}\n{'num_of_finetune:':<20}{num_of_finetune:<20}")
                    print(f"{'Model:':<20}{model:<20}")
                    print(f"{'Dataset:':<20}{dataset:<20}")
                    if with_HOSVD == 'true' or with_SVD == 'true':
                        peak_mem, total_mem, mean_mem, std_mem = get_mem(current_directory, unit=unit)
                    else:
                        mem = get_mem(current_directory, unit=unit, is_decomposition=False)
                        peak_mem, total_mem, mean_mem, std_mem = mem, float(max_epochs)*mem , mem, 0

                    if with_HOSVD == 'true':
                        print(f"{'Method:':<20}HOSVD_{explained_variance_threshold:<20}")
                        method = f"HOSVD_{explained_variance_threshold}"
                    elif with_SVD == 'true':
                        print(f"{'Method:':<20}SVD_{explained_variance_threshold:<20}")
                        method = f"SVD_{explained_variance_threshold}"
                    elif with_grad_filter == 'true':
                        print(f"{'Method:':<20}GradientFilter_{filt_radius:<20}")
                        method = f"GradientFilter_{filt_radius}"
                    
                    elif base == 'true':
                        print(f"{'Method:':<20}Vanilla BP")
                        method = "Vanilla BP"

                    print_aligned(epoch_value, val_acc, peak_mem, mean_mem, std_mem, total_mem)
                    divisor = 1 #1024*1024
                    peak_mem /= divisor
                    mean_mem /= divisor
                    std_mem /= divisor
                    total_mem /= divisor
                    results.append({
                        'Dataset': dataset,
                        'Method': method,
                        'Model': model,
                        'num_of_finetune': num_of_finetune,
                        'seed': seed,
                        'epoch_value': epoch_value,
                        'val_acc': val_acc,
                        'peak_mem': peak_mem,
                        'mean_mem': mean_mem,
                        'std_mem': std_mem,
                        'mean_mem and std_mem': f"{mean_mem:.2f} ± {std_mem:.2f}",
                        'total_mem': total_mem
                    })
                else:
                    print('No matching file found.')
                print("====================================================================================================")
            elif os.path.isdir(entry_path):
                # If it is a directory, call recursively to process subdirectories
                process_directory(entry_path)

    # Start from the root directory
    process_directory(root_directory)

    # Export results to Excel
    df = pd.DataFrame(results)
    df.to_excel('results_3.xlsx', index=False)
    print("Results have been exported to results_3.xlsx")


# Use this function
"""
root_directory: link to folder of the experiment, or event the folder of folders of many experiments

output: 'results.xlsx' file contains all the results
"""

root_directory = "runs/cls/mbv2/filt_last1_cifar10/version_10"
read_result(root_directory, unit='MB')


