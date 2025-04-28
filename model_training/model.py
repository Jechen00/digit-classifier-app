#####################################
# Packages & Dependencies
#####################################
import torch
from torch import nn


#####################################
# VGG Model Class
#####################################
class VGGBlock(nn.Module):
    '''
    Defines a modified block in the VGG architecture, 
    which includes batch normalization between convolutional layers and ReLU activations.
    
    Reference: https://poloclub.github.io/cnn-explainer/
    Reference: https://d2l.ai/chapter_convolutional-modern/vgg.html
    
    Args:
        num_convs (int): Number of consecutive convolutional layers + ReLU activations.
        in_channels (int): Number of channels in the input.
        hidden_channels (int): Number of hidden channels between convolutional layers. 
        out_channels (int): Number of channels in the output.
    '''
    def __init__(self, 
                 num_convs: int, 
                 in_channels: int, 
                 hidden_channels: int,
                 out_channels: int):
        super().__init__()
        
        self.layers = []
        
        for i in range(num_convs):
            conv_in = in_channels if i == 0 else hidden_channels
            conv_out = out_channels if i == num_convs-1 else hidden_channels
            
            self.layers += [
                nn.Conv2d(conv_in, conv_out, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm2d(conv_out),
                nn.ReLU()
            ]
    
        self.layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        self.vgg_blk = nn.Sequential(*self.layers)
            
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of VGG block.
        
        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, new_height, new_width)
        '''
        
        return self.vgg_blk(X)
    
class TinyVGG(nn.Module):
    '''
    Creates a simplified version of a VGG model, adapted from 
    https://github.com/poloclub/cnn-explainer/blob/master/tiny-vgg/tiny-vgg.py.
    The main difference is that the hidden dimensions and number of convolutional layers 
    remain the same across VGG blocks and the classifier's linear layers has output fewer features.
    
    Args:
        num_blks (int): Number of VGG blocks to put in the model
        num_convs (int): Number of consecutive convolutional layers + ReLU activations in each VGG block.
        in_channels (int): Number of channels in the input.
        hidden_channels (int): Number of hidden channels between convolutional layers. 
        fc_hidden_dim (int): Number of output (hidden) features for the first linear layer of the classifer.
        num_classes (int): Number of class labels.
        
    '''
    def __init__(self, 
                 num_blks: int, 
                 num_convs: int, 
                 in_channels: int, 
                 hidden_channels: int, 
                 fc_hidden_dim: int,
                 num_classes: int):
        super().__init__()
        
        self.all_blks = []
        for i in range(num_blks):
            conv_in = in_channels if i == 0 else hidden_channels
            self.all_blks.append(
                VGGBlock(num_convs, conv_in, hidden_channels, hidden_channels)
            )
        
        self.vgg_body = nn.Sequential(*self.all_blks)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(fc_hidden_dim), nn.ReLU(), nn.Dropout(0.5),
            nn.LazyLinear(num_classes)
        )
        
        self.vgg_body.apply(self._custom_init)
        self.classifier.apply(self._custom_init)
        
    def _custom_init(self, module):
        '''
        Initializes convolutional layer weights with Xavier initialization method.
        Initializes convolutional layer biases to zero.
        '''
        if isinstance(module, (nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
            
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the TinyVGG model.
        
        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        '''
        
        X = self.vgg_body(X)
        return self.classifier(X)
