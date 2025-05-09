#####################################
# Packages & Dependencies
#####################################
import torch
import random
import numpy as np
import os

# Setup device and multiprocessing context
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    MP_CONTEXT = None
    PIN_MEM = True
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    MP_CONTEXT = 'forkserver'
    PIN_MEM = False
else:
    DEVICE = torch.device('cpu')
    MP_CONTEXT = None
    PIN_MEM = False


#####################################
# Functions
#####################################
def set_seed(seed: int = 0):
    '''
    Sets random seed and deterministic settings for reproducibility across:
        - PyTorch
        - NumPy
        - Python's random module
    
    Args:
        seed (int): The seed value to set.
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def save_model(model: torch.nn.Module,
               save_dir: str, 
               mod_name: str):
    '''
    Saves the `state_dict()` of a model to the directory 'save_dir.'

    Args:
        model (torch.nn.Module): The PyTorch model whose state dict and keyword arguments will be saved.
        save_dir (str): Directory to save the model to.
        mod_name (str): Filename for the saved model. If this doesn't end with '.pth' or '.pt,' it will be added on for the state_dict.

    '''
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok = True)
    
    # Add .pth if it is not in mod_name
    if not mod_name.endswith('.pth') and not mod_name.endswith('.pt'):
        mod_name += '.pth'

    # Create save path
    save_path = os.path.join(save_dir, mod_name)

    # Save model's state dict
    torch.save(obj = model.state_dict(), f = save_path)