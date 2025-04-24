#####################################
# Packages
#####################################
import torch

from typing import Tuple, Dict, List
import utils

#####################################
# Functions
#####################################
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    
    '''
    Performs a training step for a PyTorch model.
    
    Args:
        model (torch.nn.Module): PyTorch model that will be trained
        dataloader (torch.utils.data.DataLoader): Dataloader containing data to train on
        loss_fn (torch.nn.Module): Loss function used as the error metric
        optimizer (torch.optim.Optimizer): Optimization method used to update model parameters per batch
        device (torch.device): Device to train on
        
    Returns:
        train_loss (float): The average loss calculated over the training set.
        train_acc (float): The accuracy calculated over the training set.
    '''
    
    model.train()
    train_loss = torch.tensor(0.0, device = device)
    train_acc = torch.tensor(0.0, device = device)
    num_samps = len(dataloader.dataset)

    # Loop through all batches in the dataloader
    for X, y in dataloader:
        
        optimizer.zero_grad() # Clear old accumulated gradients
        
        X, y = X.to(device), y.to(device)
        
        y_logits = model(X) # Get logits
        
        loss = loss_fn(y_logits, y)
        train_loss += loss.detach() * X.shape[0] # Calculate total loss for batch
        
        loss.backward() # Perform backpropagation
        optimizer.step() # Update parameters
        
        y_pred = y_logits.argmax(dim = 1) # No softmax needed for argmax (b/c preserves order)
        
        train_acc += (y_pred == y).sum() # Calculate total accuracy for batch
    
    # Get average loss and accuracy per sample
    train_loss = train_loss.item() / num_samps
    train_acc = train_acc.item() / num_samps
    
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    
    '''
    Performs a testing step for a PyTorch model.
    
    Args:
        model (torch.nn.Module): PyTorch model that will be tested.
        dataloader (torch.utils.data.DataLoader): Dataloader containing data to test on.
        loss_fn (torch.nn.Module): Loss function used as the error metric.
        device (torch.device): Device to compute on.
        
    Returns:
        test_loss (float): The average loss calculated over batches.
        test_acc (float): The average accuracy calculated over batches.
    '''
    
    model.eval()
    test_loss = torch.tensor(0.0, device = device)
    test_acc = torch.tensor(0.0, device = device)
    num_samps = len(dataloader.dataset)

    with torch.inference_mode():
        # Loop through all batches in the dataloader
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            y_logits = model(X) # Get logits

            test_loss += loss_fn(y_logits, y) * X.shape[0] # Calculate total loss for batch

            y_pred = y_logits.argmax(dim = 1) # No softmax needed for argmax (b/c preserves order)

            test_acc += (y_pred == y).sum() # Calculate total accuracy for batch

    # Get average loss and accuracy
    test_loss = test_loss.item() / num_samps
    test_acc = test_acc.item() / num_samps
    
    return test_loss, test_acc


def train(model: torch.nn.Module,
          train_dl: torch.utils.data.DataLoader,
          test_dl: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          num_epochs: int,
          patience: int,
          min_delta: float,
          device: torch.device,
          save_mod: bool = True,
          save_dir: str = '',
          mod_name: str = '') -> Dict[str, List[float]]:
    '''
    Performs the training and testing steps for a PyTorch model, 
    with early stopping applied for test loss.
    
    Args:
        model (torch.nn.Module): PyTorch model to train.
        train_dl (torch.utils.data.DataLoader): DataLoader for training.
        test_dl (torch.utils.data.DataLoader): DataLoader for testing.
        loss_fn (torch.nn.Module): Loss function used as the error metric.
        optimizer (torch.optim.Optimizer): Optimizer used to update model parameters per batch.
        
        num_epochs (int): Max number of epochs to train.
        patience (int): Number of epochs to wait before early stopping.
        min_delta (float): Minimum decrease in loss to reset counter.
        
        device (torch.device): Device to train on.
        save_mod (bool, optional): If True, saves the model after each epoch. Default is True.
        save_dir (str, optional): Directory to save the model to. Must be nonempty if save_mod is True.
        mod_name (str, optional): Filename for the saved model. Must be nonempty if save_mod is True.

    returns:
        res (dict): A results dictionary containing lists of train and test metrics for each epoch.
    '''
    
    bold_start, bold_end = '\033[1m', '\033[0m'

    if save_mod:
        assert save_dir, 'save_dir cannot be None or empty.'
        assert mod_name, 'mod_name cannot be None or empty.'

    # Initialize results dictionary
    res = {'train_loss': [],
           'train_acc': [],
           'test_loss': [],
           'test_acc': []
    }
    
    # Initialize best_loss and counter for early stopping
    best_loss, counter = None, 0
        
    for epoch in range(num_epochs):
        # Perform training and testing step
        train_loss, train_acc = train_step(model, train_dl, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_dl, loss_fn, device)
        
        # Store loss and accuracy values
        res['train_loss'].append(train_loss)
        res['train_acc'].append(train_acc)
        res['test_loss'].append(test_loss)
        res['test_acc'].append(test_acc)
        
        print(f'Epoch: {epoch + 1} | ' +
             f'train_loss = {train_loss:.4f} | train_acc = {train_acc:.4f} | ' +
             f'test_loss = {test_loss:.4f} | test_acc = {test_acc:.4f}')
        
        # Check for improvement
        if best_loss == None:
            best_loss = test_loss
            if save_mod:
                utils.save_model(model, save_dir, mod_name)

        elif test_loss < best_loss - min_delta:
            best_loss = test_loss
            counter = 0

            if save_mod:
                utils.save_model(model, save_dir, mod_name)
                print(f'{bold_start}[SAVED]{bold_end} Adequate improvement in test loss; model saved.')

        else:
            counter += 1
            if counter > patience:
                print(f'{bold_start}[ALERT]{bold_end} No improvement in test loss after {counter} epochs; early stopping triggered.')
                break

    return res