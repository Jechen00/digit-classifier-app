#####################################
# Imports & Dependencies
#####################################
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

import utils
from typing import Tuple


#####################################
# Functions
#####################################
def get_dataloaders(root: str, 
                    batch_size: int, 
                    num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    '''
    Creates training and testing dataloaders for the MNIST dataset

    Args:
        root (str): Path to download MNIST data.
        batch_size (int): Size used to split training and testing datasets into batches.
        num_workers (int): Number of workers to use for multiprocessing. Default is 0.
    '''

    # Transformations applied to each image
    transform = transforms.Compose([
        transforms.ToTensor(), # Convert to tensor and rescale pixel values to within [0, 1]
        transforms.Normalize(mean = [0.1307], std = [0.3081]) # Normalize with MNIST stats
    ])

    # Get training and testing MNIST data
    mnist_train = datasets.MNIST(root, download = True, train = True, 
                                transform = transform)
    mnist_test = datasets.MNIST(root, download = True, train = False, 
                                transform = transform)

    # Create dataloaders
    if num_workers > 0:
        mp_context = utils.MP_CONTEXT
    else:
        mp_context = None

    train_dl = DataLoader(
        dataset = mnist_train,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        multiprocessing_context = mp_context,
        pin_memory = True
    )

    test_dl = DataLoader(
        dataset = mnist_test,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        multiprocessing_context = mp_context,
        pin_memory = True
    )

    return train_dl, test_dl