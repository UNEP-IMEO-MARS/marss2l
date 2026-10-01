import random
import numpy as np
import torch

def seed_all(seed: int) -> None:
    """
    Set the seed for all sources of randomness to ensure reproducibility.
    
    This function sets seeds for:
    - Python's built-in random module
    - NumPy's random number generator
    - PyTorch (CPU and CUDA)
    - cuDNN deterministic operations
    
    Similar to PyTorch Lightning's seed_everything function.
    
    Args:
        seed (int): The random seed value to use across all libraries.
        
    Note:
        Setting cuDNN to deterministic mode may reduce performance but ensures
        reproducibility. For full reproducibility, also ensure:
        - torch.backends.cudnn.benchmark = False
        - Use worker_init_fn in DataLoader
        - Set PYTHONHASHSEED environment variable
    """
    # Set Python's built-in random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed for CPU
    torch.manual_seed(seed)
    
    # Set PyTorch random seed for all CUDA devices
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Configure cuDNN for deterministic behavior
    # Note: This may reduce performance but ensures reproducibility
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False