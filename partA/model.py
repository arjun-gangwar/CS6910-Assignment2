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
                 conv_activation: str,
                 dense_activation: str,
                 n_filters: int,
                 filter_org: str,
                 data_aug: bool,  # argparse will take, won't be required
                 batch_norm: bool,
                 dropout: float,
                 ):
        super(ConvNeuralNet, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.conv_activation = conv_activation
        self.dense_activation = dense_activation
        self.conv_activation_func = nn.ReLU()
        self.dense_activation_func = nn.ReLU()
        self.batch_norm = batch_norm
        self._init_activation()

        # compiling model
        self.conv1 = ConvBlock(activation=self.conv_activation_func,
                               batch_norm=self.batch_norm,
                               in_channels=3,
                               out_channels=16,
                               kernel_size=7,
                               padding=3,
                               stride=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = ConvBlock(activation=self.conv_activation_func,
                               batch_norm=self.batch_norm,
                               in_channels=16,
                               out_channels=32,
                               kernel_size=5,
                               padding=2,
                               stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3 = ConvBlock(activation=self.conv_activation_func,
                               batch_norm=self.batch_norm,
                               in_channels=32,
                               out_channels=64,
                               kernel_size=5,
                               padding=2,
                               stride=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv4 = ConvBlock(activation=self.conv_activation_func,
                               batch_norm=self.batch_norm,
                               in_channels=64,
                               out_channels=128,
                               kernel_size=3,
                               padding=1,
                               stride=2)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv5 = ConvBlock(activation=self.conv_activation_func,
                               batch_norm=self.batch_norm,
                               in_channels=128,
                               out_channels=256,
                               kernel_size=7,
                               padding=1,
                               stride=1)
        self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.dense1 = nn.Linear(4096, 512)

        self.dense2 = nn.Linear(512, out_dims)

        self.softmax = nn.Softmax()
        

    def _init_activation(self):
        if self.conv_activation == "relu":
            self.conv_activation_func = nn.ReLU()
        elif self.conv_activation == "softmax":
            self.conv_activation_func = nn.Softmax()
        elif self.conv_activation == "tanh":
            self.conv_activation_func = nn.Tanh()
        else:
            raise Exception("activation function not supported for convolution layer.")
        
        if self.dense_activation == "relu":
            self.dense_activation_func = nn.ReLU()
        elif self.dense_activation == "softmax":
            self.dense_activation_func = nn.Softmax()
        elif self.dense_activation == "tanh":
            self.dense_activation_func = nn.Tanh()
        else:
            raise Exception("activation function not supported for linear layer.")

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.maxpool4(x)
        x = self.conv5(x)
        x = self.maxpool5(x)
        x = x.flatten()
        x = self.dense_activation_func(self.dense1(x))
        x = self.softmax(self.dense2(x))
        return x



