'''
Define model architecture with reference to GetModel19 in ModelControl.lua
'''

from functools import partial
import torch.nn as nn
import torch

import configs
configs.init()


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()

    def forward(self, x):
        input_data = x
        x = self.blocks(x)

        # Check if narrowing is needed
        if input_data.shape[2] == x.shape[2]:
            x += input_data

        else:
            cut_input = input_data.narrow(
                2, 0, input_data.shape[2]-2*(configs.kW_first-1))
            x += self.shortcut(cut_input)

        return x


class ResNetResidualBlock_first(ResidualBlock):
    def __init__(self, in_channels, out_channels, stride=configs.dW_second, conv=partial(nn.Conv1d, kernel_size=configs.kW_first, bias=True), *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.stride, self.conv = stride, conv


class ResNetResidualBlock_second(ResidualBlock):
    def __init__(self, in_channels, out_channels, stride=configs.dW_second, conv=partial(nn.Conv1d, kernel_size=configs.kW_second, bias=True), *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.stride, self.conv = stride, conv


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):

    stacked_layers = nn.Sequential(
        conv(in_channels, out_channels, *args, **kwargs),
        nn.BatchNorm1d(out_channels)
    )

    return stacked_layers


class MainBlock_first(ResNetResidualBlock_first):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(

            conv_bn(self.in_channels, self.out_channels,
                    conv=self.conv, bias=True, stride=self.stride),

            activation(),

            conv_bn(self.out_channels, self.out_channels,
                    conv=self.conv, bias=True, stride=self.stride),

            activation()
        )


class MainBlock_second(ResNetResidualBlock_second):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(

            conv_bn(self.in_channels, self.out_channels,
                    conv=self.conv, bias=True, stride=self.stride),

            activation(),

            conv_bn(self.out_channels, self.out_channels,
                    conv=self.conv, bias=True, stride=self.stride),

            activation()
        )


class NetworkLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block, repeats=1, *args, **kwargs):
        super().__init__()

        self.blocks = nn.Sequential(
            *[block(out_channels,
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(repeats)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class Network(nn.Module):
    '''
    Stack different resnet layers of different kernel sizes
    '''
    # Kernel_sizes: [kW of first layer, kW of first chunk]

    def __init__(self, in_channels=configs.nInputPlane, n_classes=configs.categories, out_channels=configs.features,
                 kernel_sizes=[10, configs.kW_first], stride=configs.dW_first, activation=nn.ReLU,
                 block=MainBlock_first, repeats=[1, 1], *args, **kwargs,):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_classes = n_classes
        self.kernel_sizes = kernel_sizes
        self.downsample = stride
        self.repeats = repeats

        self.entry = nn.Sequential(
            nn.Conv1d(
                self.in_channels, self.out_channels, kernel_size=self.kernel_sizes[0], stride=self.downsample, bias=True),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU()
        )

        self.main_block1 = NetworkLayer(self.out_channels, self.out_channels, block=block, *args, **kwargs,
                                        activation=activation, repeats=self.repeats[0])

        self.main_block2 = NetworkLayer(self.out_channels, self.out_channels, block=MainBlock_second, *args, **kwargs,
                                        activation=activation, repeats=self.repeats[1])

        self.last = conv_bn(
            self.out_channels, self.n_classes, nn.Conv1d, kernel_size=kernel_sizes[1], stride=configs.dW_second)

    def forward(self, x):
        # might not need this since we alredy push input tensor into device in Dataset
        x = x.to(configs.device)
        # First layer
        x = self.entry(x)

        # Main Block 1 and 2
        x = self.main_block1(x)
        x = self.main_block2(x)

        # Last layer
        x = self.last(x)

        # Average on temporal dimension
        x = x.mean(2)  # Return tensor of shape (10,)

        return x


def createNetwork(depth, in_channels=configs.nInputPlane, n_classes=configs.categories):
    # Check if depth is an even number
    if depth % 2 != 0:
        print('depth must be even number')
    else:
        return Network(in_channels, n_classes, repeats=[(depth-2)//2, 12]).to(configs.device)


#model = createNetwork(depth=10)

# print(model)
