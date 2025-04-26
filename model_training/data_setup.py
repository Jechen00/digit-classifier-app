#####################################
# Packages & Dependencies
#####################################
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

import utils
from typing import Tuple

import io
import base64
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Transformations applied to each image
BASE_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(), # Convert to tensor and rescale pixel values to within [0, 1]
    transforms.Normalize(mean = [0.1307], std = [0.3081]) # Normalize with MNIST stats
])

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomAffine(degrees = 15, # Rotate up to -/+ 15 degrees
                            scale = (0.8, 1.2), # Scale between 80 and 120 percent
                            translate = (0.08, 0.08), # Translate up to -/+ 8 percent in both x and y
                            shear = 10),  # Shear up to -/+ 10 degrees
    transforms.ToTensor(), # Convert to tensor and rescale pixel values to within [0, 1]
    transforms.Normalize(mean = [0.1307], std = [0.3081]), # Normalize with MNIST stats
])


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

    # Get training and testing MNIST data
    mnist_train = datasets.MNIST(root, download = True, train = True, 
                                transform = TRAIN_TRANSFORMS)
    mnist_test = datasets.MNIST(root, download = True, train = False, 
                                transform = BASE_TRANSFORMS)

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

def mnist_preprocess(uri: str, plot: bool = False):
    '''
    Preprocesses a data URI representing a handwritten digit image according to the pipeline used in the MNIST dataset.
    The pipeline includes:
        1. Converting the image to grayscale.
        2. Resizing the image to 20x20, preserving the aspect ratio, and using anti-aliasing.
        3. Centering the resized image in a 28x28 image based on the center of mass (COM).
        4. Converting the image to a tensor (pixel values between 0 and 1) and normalizing it using MNIST statistics.

    Reference: https://paperswithcode.com/dataset/mnist

    Args:
        uri (str): A string representing the full data URI.
        plot (bool, optional): If True, the resized 20x20 image is plotted alongside the final 28x28 image (pre-normalization). 
                               The red lines on these plots intersect at the COM position. Default is False.
    Returns:
        Tensor: A tensor of shape (1, 28, 28) representing the preprocessed image, normalized using MNIST statistics.
    '''
    encoded_img = uri.split(',', 1)[1]
    image_bytes = io.BytesIO(base64.b64decode(encoded_img))
    pil_img = Image.open(image_bytes).convert('L') # Gray scale
    
    # Resize to 20x20, preserving aspect ratio, and using anti-aliasing
    pil_img.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # Convert to numpy and invert image
    img = 255 - np.array(pil_img)

    # Get image indices for y-axis (rows) and x-axis (columns)
    img_idxs = np.indices(img.shape)
    tot_mass = img.sum()
    
    # This represents the indices of the center of masses (COMs)
    com_x = np.round((img_idxs[1] * img).sum() / tot_mass).astype(int)
    com_y = np.round((img_idxs[0] * img).sum() / tot_mass).astype(int)
    
    dist_com_end_x = img.shape[1] - com_x # number of column indices from com_x to last index
    dist_com_end_y = img.shape[0] - com_y # number of row indices from com_y to last index
    
    new_img = np.zeros((28, 28), dtype = np.uint8)
    new_com_x, new_com_y = 14, 14 # Indices of the COMs for the new 28x28 image
    
    valid_start_x = min(new_com_x, com_x)
    valid_end_x = min(14, dist_com_end_x) # 14 is index distance from new COM to 28-th index
    valid_start_y = min(new_com_y, com_y)
    valid_end_y = min(14, dist_com_end_y) # 14 is index distance from new COM to 28-th index
    
    old_slice_x = slice(com_x - valid_start_x, com_x + valid_end_x)
    old_slice_y = slice(com_y - valid_start_y, com_y + valid_end_y)
    new_slice_x = slice(new_com_x - valid_start_x, new_com_x + valid_end_x)
    new_slice_y = slice(new_com_y - valid_start_y, new_com_y + valid_end_y)

    # Paste cropped image into 28x28 field such that the old COM (com_y, com_x), is at the center (14, 14)
    new_img[new_slice_y, new_slice_x] = img[old_slice_y, old_slice_x]
    
    if plot:
        fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 6))

        axes[0].imshow(img, cmap = 'grey')
        axes[0].axhline(com_y, c = 'red')
        axes[0].axvline(com_x, c = 'red')

        axes[1].imshow(new_img, cmap = 'grey')
        axes[1].axhline(new_com_y, c = 'red')
        axes[1].axvline(new_com_x, c = 'red')
        
        axes[0].set_title(f'Original Resized {img.shape[0]}x{img.shape[1]} Image')
        axes[1].set_title('New Centered 28x28 Image')
        
        plt.tight_layout()

    # Return transformed tensor of new image. This includes normalizing to MNIST stats
    return BASE_TRANSFORMS(new_img)