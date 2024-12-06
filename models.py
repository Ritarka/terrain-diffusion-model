# CMU 18-794 Homework 4
# The code base is based on the great work from CSC 321, U Toronto
# https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-code.zip
# CSC 321, Assignment 4
#
# This file contains the models used for both parts of the assignment:
#
#   - DCGenerator        --> Used in the vanilla GAN in Part 1
#   - DCDiscriminator    --> Used in both the vanilla GAN in Part 1
# For the assignment, you are asked to create the architectures of these
# three networks by filling in the __init__ and forward methods in the
# DCGenerator, DCDiscriminator classes.
# Feel free to add and try your own models

import torch
import torch.nn as nn


def up_conv(in_channels, out_channels, kernel_size, stride=1, padding=1,
            scale_factor=2, norm='batch', activ=None):
    """Create a transposed-convolutional layer, with optional normalization."""
    layers = []
    layers.append(nn.Upsample(scale_factor=scale_factor, mode='nearest'))
    layers.append(nn.Conv2d(
        in_channels, out_channels,
        kernel_size, stride, padding, bias=norm is None
    ))
    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))

    if activ == 'relu':
        layers.append(nn.ReLU())
    elif activ == 'leaky':
        layers.append(nn.LeakyReLU())
    elif activ == 'tanh':
        layers.append(nn.Tanh())

    return nn.Sequential(*layers)

def down_conv(in_channels, out_channels, kernel_size, stride=1, padding=1, norm='batch'):
    """Create a transposed-convolutional layer, with optional normalization."""
    layers = []
    layers.append(nn.Conv2d(
        in_channels, out_channels,
        kernel_size, stride, padding, bias=norm is None
    ))
    layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU())

    return nn.Sequential(*layers)


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1,
         norm='batch', init_zero_weights=False, activ=None):
    """Create a convolutional layer, with optional normalization."""
    layers = []
    conv_layer = nn.Conv2d(
        in_channels=in_channels, out_channels=out_channels,
        kernel_size=kernel_size, stride=stride, padding=padding,
        bias=norm is None
    )
    if init_zero_weights:
        conv_layer.weight.data = 0.001 * torch.randn(
            out_channels, in_channels, kernel_size, kernel_size
        )
    layers.append(conv_layer)

    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))

    if activ == 'relu':
        layers.append(nn.ReLU())
    elif activ == 'leaky':
        layers.append(nn.LeakyReLU())
    elif activ == 'tanh':
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


# class DCGenerator(nn.Module):

class DCGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=32):
        super(DCGenerator, self).__init__()

        # Encoder layers
        self.e1 = nn.Conv2d(input_nc, ngf, kernel_size=3, stride=1, padding=1)
        self.e2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
        )
        self.e3 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
        )
        self.e4 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
        )
        self.e5 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, ngf // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf // 2),
        )
        self.e6 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf // 2, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
        )

        # Decoder layers
        self.d1_ = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(1, ngf // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf // 2),
        )
        self.d2_ = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf // 2, ngf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
        )
        self.d3_ = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
        )
        self.d4_ = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
        )
        self.d5_ = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
        )
        self.d61 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf, output_nc, kernel_size=3, stride=1, padding=1),
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Encoder forward pass
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)

        # Decoder forward pass with skip connections
        d1_ = self.d1_(e6)
        d2_ = self.d2_(d1_)
        d2 = d2_ + e4  # Skip connection

        d3_ = self.d3_(d2)
        d4_ = self.d4_(d3_)
        d4 = d4_ + e2  # Skip connection

        d5_ = self.d5_(d4)
        d61 = self.d61(d5_)
        out = self.tanh(d61)

        return out


class ResnetBlock(nn.Module):

    def __init__(self, conv_dim, norm, activ):
        super().__init__()
        self.conv_layer = conv(
            in_channels=conv_dim, out_channels=conv_dim,
            kernel_size=3, stride=1, padding=1, norm=norm,
            activ=activ
        )

    def forward(self, x):
        out = x + self.conv_layer(x)
        return out


class DCDiscriminator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ndf=64, n_layers=3):
        super(DCDiscriminator, self).__init__()
        layers = []

        # Initial Convolution
        layers.append(nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Intermediate Layers
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            layers.append(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(ndf * nf_mult))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Final Layer before flattening
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        layers.append(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(ndf * nf_mult))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Final Convolution to produce single-channel output
        layers.append(nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1))

        # Flatten and Output Layer
        self.model = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(900, 1)  # Map 1x1 feature map to a scalar value
        self.sigmoid = nn.Sigmoid()  # Output probabilities between 0 and 1

    def forward(self, x):
        x = self.model(x)
        x = self.flatten(x)  # Flatten 1x1 feature map
        x = self.fc(x)  # Fully connected layer
        return self.sigmoid(x)
