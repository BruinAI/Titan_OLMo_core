import time
import torch
from functools import wraps
import os
import psutil

memory_log_file = "memory_profile.log"

def log_memory(message, clear_file=False):
    """Write memory usage information to log file"""
    mode = "w" if clear_file else "a"
    with open(memory_log_file, mode) as f:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Get GPU memory if available
        gpu_allocated = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
        gpu_reserved = torch.cuda.memory_reserved() / (1024**3) if torch.cuda.is_available() else 0
        
        # Get CPU memory
        process = psutil.Process(os.getpid())
        cpu_memory = process.memory_info().rss / (1024**3)
        
        log_line = f"[{timestamp}] {message} - GPU Allocated: {gpu_allocated:.4f}GB, Reserved: {gpu_reserved:.4f}GB, CPU: {cpu_memory:.4f}GB\n"
        f.write(log_line)
        print(log_line.strip())

def reset_peak_stats():
    """Reset peak memory stats for better measuring individual operations"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def measure_memory(func=None, name=None):
    """Decorator to measure memory usage before and after function execution"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = name or func.__name__
            reset_peak_stats()
            
            # Log before execution
            log_memory(f"BEFORE {func_name}")
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Log after execution with peak memory
            peak = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
            log_memory(f"AFTER {func_name} - Peak during execution: {peak:.4f}GB")
            
            return result
        return wrapper
    
    if func is None:
        return decorator
    return decorator(func)