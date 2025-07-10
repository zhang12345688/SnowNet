from typing import Optional, Union, Sequence
from mmengine.model import BaseModule, constant_init
from mmcv.cnn import ConvModule, build_norm_layer
from mmcv.cnn.bricks import DropPath
import torch
import torch.nn as nn

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

def make_divisible(value, divisor, min_value=None, min_ratio=0.9):
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value

class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

class CAR(BaseModule):
    def __init__(
            self,
            channels0: int,
            channels: int,
            h_kernel_size: int = 11,
            v_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.depth_conv1 = ConvModule(channels0, channels, 3, 1, 1,
                                      groups=channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.pointwise_conv1 = ConvModule(channels, channels, 1, 1, 0,
                                          norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.h_conv = ConvModule(channels, channels, (1, h_kernel_size), 1,
                                       (0, h_kernel_size // 2), groups=channels,
                                       norm_cfg=None, act_cfg=None)
        self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1), 1,
                                       (v_kernel_size // 2, 0), groups=channels,
                                       norm_cfg=None, act_cfg=None)
        self.depth_conv2 = ConvModule(channels, channels, 3, 1, 1,
                                      groups=channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.pointwise_conv2 = ConvModule(channels, channels, 1, 1, 0,
                                          norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        residual = x
        x = self.avg_pool(x)
        x = self.pointwise_conv1(self.depth_conv1(x))
        x = self.h_conv(x)
        x = self.v_conv(x)
        x = self.pointwise_conv2(self.depth_conv2(x))
        x = x + residual
        return x

class HFA(nn.Module):
    def __init__(self,
                 channels1, channels,
                 kernel_sizes=[5, [1, 9], [1, 13], [1, 25]],
                 paddings=[2, [0, 4], [0, 6], [0, 12]]):
        super().__init__()
        self.depthwise0 = nn.Conv2d(
            channels1,
            channels,
            kernel_size=kernel_sizes[0],
            padding=paddings[0],
            groups=channels)
        self.pointwise0 = nn.Conv2d(channels, channels, kernel_size=1)
        for i, (kernel_size, padding) in enumerate(zip(kernel_sizes[1:], paddings[1:])):
            kernel_size_ = [kernel_size, kernel_size[::-1]]
            padding_ = [padding, padding[::-1]]
            conv_name = [f'depthwise{i}_1', f'depthwise{i}_2', f'pointwise{i}_1', f'pointwise{i}_2']
            for i_kernel, i_pad, i_conv in zip(kernel_size_, padding_, conv_name[:2]):
                self.add_module(
                    i_conv,
                    nn.Conv2d(
                        channels,
                        channels,
                        tuple(i_kernel),
                        padding=i_pad,
                        groups=channels))
            for i_conv in conv_name[2:]:
                self.add_module(i_conv, nn.Conv2d(channels, channels, kernel_size=1))
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=1)
        self.branch4 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False),
            nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=1, padding=(0, 1), groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=(3, 1), stride=1, padding=(1, 0), groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=5, dilation=5, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False),
            nn.Conv2d(channels, channels, kernel_size=(3, 1), stride=1, padding=(1, 0), groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=1, padding=(0, 1), groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=5, dilation=5, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False),
        )
        self.conv4 = nn.Conv2d(3 * channels, channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        u = x.clone()
        attn = self.pointwise0(self.depthwise0(x))
        attn_0 = self.depthwise0_2(self.depthwise0_1(attn))
        attn_1 = self.depthwise1_2(self.depthwise1_1(attn))
        attn_2 = self.depthwise2_2(self.depthwise2_1(attn))
        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)
        x3 = attn * u
        x4 = self.branch4(x)
        x5 = self.branch5(x)
        out = torch.cat((x4, x5, x3), 1)
        attn1 = self.conv4(out)
        x4 = attn1 * u
        return x4

class LKRBlock(BaseModule):
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            expansion: float = 1.0,
            drop_path_rate: float = 0.,
            layer_scale: Optional[float] = 1.0,
            add_identity: bool = True,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        out_channels = out_channels or in_channels
        hidden_channels = make_divisible(int(out_channels * expansion), 8)
        self.block = HFA(hidden_channels, hidden_channels)
        self.caa = CAR(hidden_channels, hidden_channels)
        self.add_identity = add_identity and in_channels == out_channels
        self.pre_conv = ConvModule(in_channels, hidden_channels, 1, 1, 0, 1,
                                   norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.post_conv = ConvModule(hidden_channels, out_channels, 1, 1, 0, 1,
                                    norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.caa_factor = CAR(hidden_channels, hidden_channels)

    def forward(self, x):
        x = self.pre_conv(x)
        y = x
        x = self.block(x)
        if self.caa_factor is not None:
            y = self.caa(y)
        if self.add_identity:
            y = x * y
            x = x + y
        else:
            x = x * y
        x = self.post_conv(x)
        return x