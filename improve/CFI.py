import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvFFN(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=4):
        super(ConvFFN, self).__init__()
        internal_channels = in_channels * expand_ratio
        self.dw1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels,
                             bias=False)
        self.pw1 = nn.Conv2d(in_channels, internal_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.dw2 = nn.Conv2d(internal_channels, internal_channels, kernel_size=3, stride=1, padding=1,
                             groups=internal_channels, bias=False)
        self.pw2 = nn.Conv2d(internal_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.nonlinear = nn.SiLU()

    def forward(self, x):
        x1 = self.dw1(x)
        x2 = self.pw1(x1)
        x3 = self.nonlinear(x2)
        x4 = self.dw2(x3)
        x5 = self.pw2(x4)
        x6 = self.nonlinear(x5)
        return x6 + x


class CFIBlock(nn.Module):
    def __init__(self, in_channel, out_channel, ratio=1):
        super(CFIBlock, self).__init__()
        internal_channel = in_channel * ratio
        self.relu = nn.SiLU()

        self.dw_13 = nn.Conv2d(internal_channel, internal_channel, kernel_size=(1, 7), padding=(0, 3), stride=1,
                               groups=internal_channel, bias=False)
        self.dw_31 = nn.Conv2d(internal_channel, internal_channel, kernel_size=(7, 1), padding=(3, 0), stride=1,
                               groups=internal_channel, bias=False)
        self.dw_33 = nn.Conv2d(internal_channel, internal_channel, kernel_size=7, padding=3, stride=1,
                               groups=internal_channel, bias=False)
        self.dw_11 = nn.Conv2d(internal_channel, internal_channel, kernel_size=1, padding=0, stride=1,
                               groups=internal_channel, bias=False)

        #self.pw1 = nn.Conv2d(internal_channel, internal_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.convFFN = ConvFFN(in_channels=in_channel, out_channels=in_channel)
        self.batch_norm = nn.BatchNorm2d(in_channel)
        self.pw1_1 = nn.Conv2d(in_channel, internal_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.pw2 = nn.Conv2d(internal_channel, in_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.c1 = nn.Conv2d(4 * internal_channel,internal_channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x1 = self.pw1_1(x)
        x1 = self.relu(x1)
        x2 = self.dw_13(x1)
        x3 = self.dw_31(x1)
        x5 = self.dw_33(x1)
        x6 = self.dw_11(x1)
        x7 = self.c1(torch.cat((x2, x3, x5, x6), 1))
        # x7 = self.pw1(x7)
        # x7 = self.relu(x7)
        x7 = self.pw2(x7)
        x7 = self.relu(x7)
        x7 = self.batch_norm(x7)
        x4 = self.convFFN(x7)
        return x4 + x