import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, 
                 activation, 
                 batch_norm,
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class ConvNeuralNet(nn.Module):
    def __init__(self,
                 in_dims, 
                 out_dims,
                 activation: str,
                 n_filters: list,
                 filter_org: str,
                 data_aug: bool,  # argparse will take, won't be required
                 batch_norm: bool,
                 dropout: int,
                 ):
        super(ConvNeuralNet, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.activation = activation
        self.activation_func = None
        self.batch_norm = batch_norm
        self._init_activation()

        # compiling model
        self.conv1 = ConvBlock(activation=self.activation_func,
                               batch_norm=self.batch_norm,
                               in_channels=self.in_dims,
                               out_channels=16,
                               kernel_size=7)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3)

        self.conv2 = ConvBlock(activation=self.activation_func,
                               batch_norm=self.batch_norm,
                               in_channels=self.in_dims,
                               out_channels=16,
                               kernel_size=7)
        self.maxpool2 = nn.MaxPool2d()

        self.conv3 = ConvBlock(activation=self.activation_func,
                               batch_norm=self.batch_norm,
                               in_channels=self.in_dims,
                               out_channels=16,
                               kernel_size=7)
        self.maxpool3 = nn.MaxPool2d()

        self.conv4 = ConvBlock(activation=self.activation_func,
                               batch_norm=self.batch_norm,
                               in_channels=self.in_dims,
                               out_channels=16,
                               kernel_size=7)
        self.maxpool4 = nn.MaxPool2d()

        self.conv5 = ConvBlock(activation=self.activation_func,
                               batch_norm=self.batch_norm,
                               in_channels=self.in_dims,
                               out_channels=16,
                               kernel_size=7)
        self.maxpool5 = nn.MaxPool2d()

        self.dense = nn.Linear()
        self.softmax = nn.Softmax()
        

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

