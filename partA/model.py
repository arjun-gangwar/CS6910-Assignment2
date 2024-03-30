import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, 
                 activation, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.activation = activation
    def forward(self, x):
        return self.activation(self.conv(x))

class ConvNeuralNet(nn.Module):
    def __init__(self, 
                 activation: str,
                 n_filters: list,
                 filter_org: str,
                 data_aug: bool,  # argparse will take, won't be required
                 batch_norm: bool,
                 dropout: int,
                 ):
        super(ConvNeuralNet, self).__init__()
        self.activation = activation
        self.activation_func = None
        self._init_activation()

    def _init_activation(self):
        if self.activation == "relu":
            self.activation_func = nn.ReLU()
        elif self.activation == "softmax":
            self.activation_func = nn.Softmax()
        elif self.activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise Exception("activation function not supported.")

    def forward(self, x):
        pass

