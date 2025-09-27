import torch
import os
from typing import Tuple

def get_device() -> Tuple[torch.device, bool]:
    """Get the best available device and whether we're using CPU."""
    using_cpu = False
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        using_cpu = True
        device = torch.device('cpu')
    
    return device, using_cpu

def get_optimal_workers(using_cpu: bool) -> Tuple[int, int]:
    """Get optimal number of workers for train and validation loaders."""
    if using_cpu:
        return 1, 1
    
    num_cpus = os.cpu_count()
    worker_num_unit = (num_cpus if num_cpus else 1) // 3
    train_workers = max(1, worker_num_unit * 2)
    val_workers = max(1, worker_num_unit)
    
    return train_workers, val_workers