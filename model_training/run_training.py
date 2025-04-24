#####################################
# Packages & Dependencies
#####################################
import argparse
import torch
from torch import nn

import utils, data_setup, model, engine

# Setup random seeds
utils.set_seed(6)

# Setup hyperparameters
parser = argparse.ArgumentParser()

parser.add_argument('-nw', '--num_workers', help = 'Number of workers for dataloaders.',
                    type = int, default = 0)
parser.add_argument('-ne', '--num_epochs', help = 'Number of epochs to train model for.', 
                    type = int, default = 15)
parser.add_argument('-bs', '--batch_size', help = 'Size of batches to split training set.',
                    type = int, default = 64)
parser.add_argument('-lr', '--learning_rate', help = 'Learning rate for the optimizer.', 
                    type = float, default = 0.01)
parser.add_argument('-p', '--patience', help = 'Number of epochs to wait before early stopping.', 
                    type = int, default = 5)
parser.add_argument('-md', '--min_delta', help = 'Minimum decrease in loss to reset patience.', 
                    type = float, default = 0.001)

args = parser.parse_args()


#####################################
# Training Code
#####################################
if __name__ == '__main__':

    print(f'{'#' * 50}\n'
          f'\033[1mTraining hyperparameters:\033[0m \n'
          f'    - num_workers:   {args.num_workers} \n'
          f'    - num_epochs:    {args.num_epochs} \n'
          f'    - batch_size:    {args.batch_size} \n'
          f'    - learning_rate: {args.learning_rate} \n'
          f'    - patience:      {args.patience} \n'
          f'    - min_delta:     {args.min_delta} \n'
          f'{'#' * 50}')

    # Get dataloaders
    train_dl, test_dl = data_setup.get_dataloaders(root = '../mnist_data',
                                                   batch_size = args.batch_size,
                                                   num_workers = args.num_workers)
    
    # Set up saving directory and file name
    save_dir = '../models'
    mod_name = 'tiny_vgg_model.pth'

    # Get TinyVGG model
    mod_kwargs = {
        'num_blks': 2,
        'num_convs': 2,
        'in_channels': 1,
        'hidden_channels': 10,
        'num_classes': len(train_dl.dataset.classes)
    }
    vgg_mod = model.TinyVGG(**mod_kwargs).to(utils.DEVICE)

    # Get loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params = vgg_mod.parameters(), lr = args.learning_rate)

    # Train model
    mod_res = engine.train(model = vgg_mod,
                           train_dl = train_dl,
                           test_dl = test_dl,
                           loss_fn = loss_fn,
                           optimizer = optimizer,
                           num_epochs = args.num_epochs,
                           patience = args.patience,
                           min_delta = args.min_delta,
                           device = utils.DEVICE,
                           save_mod = True,
                           save_dir = save_dir,
                           mod_name = mod_name)