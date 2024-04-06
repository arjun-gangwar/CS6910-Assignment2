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
                 dense_size: int,
                 filter_size: list,
                 n_filters: int,
                 filter_org: str,
                 batch_norm: bool,
                 dropout: float,
                 ):
        super(ConvNeuralNet, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.conv_activation = conv_activation
        self.dense_activation = dense_activation
        self.dense_size = dense_size
        self.filter_size = filter_size
        self.n_filters = n_filters
        self.filter_org = filter_org
        self.filter_counts = None
        self.conv_activation_func = nn.ReLU()
        self.dense_activation_func = nn.ReLU()
        self.batch_norm = batch_norm
        self.dropout = nn.Dropout(dropout)
        self._init_activation()
        self._init_filter_counts() 

        # compiling model
        self.conv1 = ConvBlock(activation=self.conv_activation_func,
                               batch_norm=self.batch_norm,
                               in_channels=3,
                               out_channels=self.filter_counts[0],
                               kernel_size=self.filter_size[0],
                               padding="same",
                               stride=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = ConvBlock(activation=self.conv_activation_func,
                               batch_norm=self.batch_norm,
                               in_channels=self.filter_counts[0],
                               out_channels=self.filter_counts[1],
                               kernel_size=self.filter_size[1],
                               padding="same",
                               stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3 = ConvBlock(activation=self.conv_activation_func,
                               batch_norm=self.batch_norm,
                               in_channels=self.filter_counts[1],
                               out_channels=self.filter_counts[2],
                               kernel_size=self.filter_size[2],
                               padding="same",
                               stride=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv4 = ConvBlock(activation=self.conv_activation_func,
                               batch_norm=self.batch_norm,
                               in_channels=self.filter_counts[2],
                               out_channels=self.filter_counts[3],
                               kernel_size=self.filter_size[3],
                               padding="same",
                               stride=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv5 = ConvBlock(activation=self.conv_activation_func,
                               batch_norm=self.batch_norm,
                               in_channels=self.filter_counts[3],
                               out_channels=self.filter_counts[4],
                               kernel_size=self.filter_size[4],
                               padding="same",
                               stride=1)
        self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.dense1 = nn.Linear(self.filter_counts[4] * 8 * 8, self.dense_size)

        self.dense2 = nn.Linear(self.dense_size, out_dims)

        self.softmax = nn.Softmax(dim=1)

    def _init_activation(self):
        if self.conv_activation == "relu":
            self.conv_activation_func = nn.ReLU()
        elif self.conv_activation == "gelu":
            self.conv_activation_func = nn.GELU()
        elif self.conv_activation == "silu":
            self.conv_activation_func = nn.SiLU()
        elif self.conv_activation == "mish":
            self.conv_activation_func = nn.Mish()
        else:
            raise Exception("activation function not supported for convolution layer.")
        
        if self.dense_activation == "relu":
            self.dense_activation_func = nn.ReLU()
        elif self.dense_activation == "sigmoid":
            self.dense_activation_func = nn.Sigmoid()
        elif self.dense_activation == "tanh":
            self.dense_activation_func = nn.Tanh()
        else:
            raise Exception("activation function not supported for linear layer.")
        
    def _init_filter_counts(self):
        num_layers=5
        if self.n_filters < 16:
            raise Exception("number of filters in the first layer cann't be less than 16")
        else:
            if self.filter_org == "same":
                self.filter_counts = [self.n_filters] * num_layers
            elif self.filter_org == "double":
                self.filter_counts = [self.n_filters]
                for _ in range(num_layers-1):
                    self.filter_counts.append(self.filter_counts[-1] * 2)
            elif self.filter_org == "halve":
                self.filter_counts = [self.n_filters]
                for _ in range(num_layers-1):
                    self.filter_counts.append(self.filter_counts[-1] // 2)
            else:
                raise Exception("filter organization not supported.")

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
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(self.dense_activation_func(self.dense1(x)))
        x = self.softmax(self.dense2(x))
        return x



