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
        self.batch_norm = batch_norm
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        if self.batch_norm:
            return self.activation(self.bn(self.conv(x)))
        return self.activation(self.conv(x))

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
                               kernel_size=7,
                               padding=3,
                               stride=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = ConvBlock(activation=self.activation_func,
                               batch_norm=self.batch_norm,
                               in_channels=self.in_dims,
                               out_channels=32,
                               kernel_size=5,
                               padding=2,
                               stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3 = ConvBlock(activation=self.activation_func,
                               batch_norm=self.batch_norm,
                               in_channels=self.in_dims,
                               out_channels=64,
                               kernel_size=5,
                               padding=2,
                               stride=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv4 = ConvBlock(activation=self.activation_func,
                               batch_norm=self.batch_norm,
                               in_channels=self.in_dims,
                               out_channels=128,
                               kernel_size=3,
                               padding=1,
                               stride=2)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv5 = ConvBlock(activation=self.activation_func,
                               batch_norm=self.batch_norm,
                               in_channels=self.in_dims,
                               out_channels=256,
                               kernel_size=7,
                               padding=1,
                               stride=1)
        self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.dense = nn.Linear(8*8*256)
        self.softmax = nn.Softmax()

        self.loss = nn.CrossEntropyLoss()
        

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

