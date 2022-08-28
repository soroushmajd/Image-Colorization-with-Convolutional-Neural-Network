######################################################################
# MODELS
######################################################################
import torch.nn as nn
import torch
from torchsummary import summary
import torch.nn.functional as F

import math
class MyConv2d(nn.Module):
    """
    Our simplified implemented of nn.Conv2d module for 2D convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=None):
        super(MyConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if padding is None:
            self.padding = kernel_size // 2
        else:
            self.padding = padding
        self.weight = nn.parameter.Parameter(torch.Tensor(
            out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.parameter.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels * self.kernel_size * self.kernel_size
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, padding=self.padding)


class CNN(nn.Module):
    def __init__(self, kernel, num_filters, num_colours, num_in_channels):
        super(CNN, self).__init__()
        padding = kernel // 2

        self.firstLayer = nn.Sequential(
            MyConv2d(num_in_channels, num_filters, kernel, padding=padding),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )

        self.secondLayer = nn.Sequential(
            MyConv2d(num_filters, 2*num_filters, kernel, padding=padding),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(2*num_filters),
            nn.ReLU()
        )

        self.thirdLayer = nn.Sequential(
            MyConv2d(2*num_filters, 2*num_filters, kernel, padding=padding),
            nn.BatchNorm2d(2*num_filters),
            nn.ReLU()
        )

        self.fourthLayer = nn.Sequential(
            MyConv2d(2*num_filters, num_filters, kernel, padding=padding),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )

        self.fifthLayer = nn.Sequential(
            MyConv2d(num_filters, num_colours, kernel, padding=padding),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(num_colours),
            nn.ReLU()
        )

        self.sixthLayer = MyConv2d(num_colours, num_colours, kernel, padding=padding)

    def forward(self, x):
        first = self.firstLayer(x)
        second = self.secondLayer(first)
        third = self.thirdLayer(second)
        fourth = self.fourthLayer(third)
        fifth = self.fifthLayer(fourth)
        output = self.sixthLayer(fifth)
        return output


class UNet(nn.Module):
    def __init__(self, kernel, num_filters, num_colours, num_in_channels):
        super(UNet, self).__init__()

        self.firstLayer = nn.Sequential(
            MyConv2d(num_in_channels, num_filters, kernel),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )

        self.secondLayer = nn.Sequential(
            MyConv2d(num_filters, 2*num_filters, kernel),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(2*num_filters),
            nn.ReLU()
        )

        self.thirdLayer = nn.Sequential(
            MyConv2d(2*num_filters, 2*num_filters, kernel),
            nn.BatchNorm2d(2*num_filters),
            nn.ReLU()
        )

        self.fourthLayer = nn.Sequential(
            MyConv2d(2*2*num_filters, num_filters, kernel),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )

        self.fifthLayer = nn.Sequential(
            MyConv2d(2*num_filters, num_colours, kernel),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(num_colours),
            nn.ReLU()
        )

        self.sixthLayer = MyConv2d(num_colours+num_in_channels, num_colours, kernel)

    def forward(self, x):
        first = self.firstLayer(x)
        second = self.secondLayer(first)
        third = self.thirdLayer(second)
        fourth = self.fourthLayer(torch.cat([second, third], dim=1))
        fifth = self.fifthLayer(torch.cat([first, fourth], dim=1))
        sixth = self.sixthLayer(torch.cat([x, fifth], dim=1))
        return sixth


class SkipConnection(nn.Module):
    def __init__(self, in_channels, out_channels, enc_or_dec='encoder'):
        super(SkipConnection, self).__init__()
        self.enc_or_dec = enc_or_dec
        self.skip_encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=2, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

        self.skip_decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        if self.enc_or_dec == 'encoder':
            return self.skip_encoder(x)
        if self.enc_or_dec == 'decoder':
            return self.skip_decoder(x)


class CustomUnetWithResiduals(nn.Module):
    def __init__(self, kernel, num_filters, num_colours, num_in_channels):
        super(CustomUnetWithResiduals, self).__init__()

        self.firstLayer = nn.Sequential(
            MyConv2d(num_in_channels, num_filters, kernel),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )
        self.skip_first_layer = SkipConnection(num_in_channels, num_filters, enc_or_dec='encoder')
       

        self.secondLayer = nn.Sequential(
            MyConv2d(num_filters, 2*num_filters, kernel),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(2*num_filters),
            nn.ReLU()
        )
        self.skip_second_layer = SkipConnection(num_filters, num_filters * 2, enc_or_dec='encoder')

        self.thirdLayer = nn.Sequential(
            MyConv2d(2*num_filters, 2*num_filters, kernel),
            nn.BatchNorm2d(2*num_filters),
            nn.ReLU()
        )

        self.fourthLayer = nn.Sequential(
            MyConv2d(2*2*num_filters, num_filters, kernel),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )
        self.skip_fourth_layer = SkipConnection(num_filters * 2 * 2, num_filters, enc_or_dec='decoder')


        self.fifthLayer = nn.Sequential(
            MyConv2d(2*num_filters, num_colours, kernel),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(num_colours),
            nn.ReLU()
        )
        self.skip_fifth_layer = SkipConnection(num_filters * 2, num_colours, enc_or_dec='decoder')

        self.sixthLayer = MyConv2d(num_colours+num_in_channels, num_colours, kernel)

    def forward(self, x):
        first = self.firstLayer(x)
        skip = self.skip_first_layer(x)
        first = first + skip

        second = self.secondLayer(first)
        skip = self.skip_second_layer(first)
        second = second + skip

        third = self.thirdLayer(second)
        third = third + second

        fourth = self.fourthLayer(torch.cat([second, third], dim=1))
        skip = self.skip_fourth_layer(torch.cat([second, third], dim=1))
        fourth = fourth + skip

        fifth = self.fifthLayer(torch.cat([first, fourth], dim=1))
        skip = self.skip_fifth_layer(torch.cat([first, fourth], dim=1))
        fifth = fifth + skip

        sixth = self.sixthLayer(torch.cat([x, fifth], dim=1))
        return sixth

