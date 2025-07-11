import torch
import torch.nn as nn
class ConvBlock(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=2,
                 padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))


class LFEM(torch.nn.Module):
    def __init__(self,in_channels,out_channels ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels=out_channels
        self.convblock = ConvBlock(in_channels=self.in_channels,
                                   out_channels=self.out_channels,
                                   stride=1)
        self.conv1 = nn.Conv2d(out_channels,out_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels,out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        feature = self.convblock(x)
        x = self.avgpool(feature)

        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x

class SFRM(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input):
        # global average pooling
        x = self.avgpool(input)
        assert self.in_channels == x.size(
            1), 'in_channels and out_channels should all be {}'.format(
                x.size(1))
        x = self.conv(x)
        # x = self.sigmoid(self.bn(x))
        x = self.sigmoid(x)
        # channels of input and x should be same
        x = torch.mul(input, x)
        return x

class MFFM(torch.nn.Module):
    def __init__(self,in_channels,out_channels ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels=out_channels
        self.convblock = ConvBlock(in_channels=self.in_channels,
                                   out_channels=self.out_channels,
                                   stride=1)
        self.conv1 = nn.Conv2d(out_channels,out_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels,out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        feature1 = self.convblock(x)
        feature=self.conv3(feature1)
        x = self.avgpool(feature)
        x1=self.conv4(feature)
        x = torch.add(x1, x)
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature1, x)
        return x
