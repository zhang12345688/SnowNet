import numpy as np
import torch
import torch.nn as nn

from nets.backbone import Backbone, C2f, Conv
from nets.yolo_training import weights_init
from utils.utils_bbox import make_anchors

from improve.deform import DeformConv2d
from improve.MAR import LFEM, MFFM, SFRM
from improve.CFI import CFIBlock

def fuse_conv_and_bn(conv, bn):
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          dilation=conv.dilation,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fusedconv

class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.BatchNorm2d(in_features),
                      nn.SiLU(),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.BatchNorm2d(in_features)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Decoder(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(Decoder, self).__init__()
        wid_mul = 0.50
        base_channels = int(wid_mul * 64)
        deep_mul = 1.00
        self.conv_1 = nn.Sequential(
            nn.ConvTranspose2d(int(base_channels * 16 * deep_mul), base_channels * 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.SiLU()
        )
        self.r1 = ResidualBlock(base_channels * 8)
        self.d1 = DeformConv2d(base_channels * 8, base_channels * 8)
        self.f1 = LFEM(base_channels * 8, base_channels * 8)
        self.conv_2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 16, base_channels * 4, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.SiLU()
        )
        self.r2 = ResidualBlock(base_channels * 4)
        self.d2 = DeformConv2d(base_channels * 4, base_channels * 4)
        self.f2 = MFFM(base_channels * 4, base_channels * 4)
        self.conv_3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_channels),
            nn.SiLU()
        )
        self.r3 = ResidualBlock(base_channels)
        self.f3 = SFRM(base_channels, base_channels)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv_4 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_channels, output_nc, 7),
            nn.Tanh()
        )

    def forward(self, x, y, z):
        c1 = self.conv_1(x)
        c1 = self.r1(c1)
        c1 = self.d1(c1)
        c1 = self.f1(c1)
        skip1_de = torch.cat((c1, y), 1)
        c2 = self.conv_2(skip1_de)
        c2 = self.r2(c2)
        c2 = self.d2(c2)
        c2 = self.f2(c2)
        skip2_de = torch.cat((c2, z), 1)
        c3 = self.conv_3(skip2_de)
        c3 = self.r3(c3)
        c3 = self.f3(c3)
        c4 = self.upsample(c3)
        dehaze = self.conv_4(c4)
        return dehaze

class YoloBody(nn.Module):
    def __init__(self, input_shape, num_classes, phi, pretrained=False):
        super(YoloBody, self).__init__()
        depth_dict = {'n': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.00}
        width_dict = {'n': 0.25, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25}
        deep_width_dict = {'n': 1.00, 's': 1.00, 'm': 0.75, 'l': 0.50, 'x': 0.50}
        dep_mul, wid_mul, deep_mul = depth_dict[phi], width_dict[phi], deep_width_dict[phi]
        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)
        self.backbone = Backbone(base_channels, base_depth, deep_mul, phi, pretrained=pretrained)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv3_for_upsample1 = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8, base_channels * 8, base_depth, shortcut=False)
        self.conv3_for_upsample2 = C2f(base_channels * 8 + base_channels * 4, base_channels * 4, base_depth, shortcut=False)
        self.down_sample1 = Conv(base_channels * 4, base_channels * 4, 3, 2)
        self.conv3_for_downsample1 = C2f(base_channels * 8 + base_channels * 4, base_channels * 8, base_depth, shortcut=False)
        self.down_sample2 = Conv(base_channels * 8, base_channels * 8, 3, 2)
        self.conv3_for_downsample2 = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8, int(base_channels * 16 * deep_mul), base_depth, shortcut=False)
        self.i1 = CFIBlock(base_channels * 4, base_channels * 4)
        self.i2 = CFIBlock(base_channels * 8, base_channels * 8)
        self.i3 = CFIBlock(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul))
        self.decoder = Decoder(int(base_channels * 16 * deep_mul), 3)
        ch = [base_channels * 4, base_channels * 8, int(base_channels * 16 * deep_mul)]
        self.shape = None
        self.nl = len(ch)
        self.stride = torch.tensor([256 / x.shape[-2] for x in self.backbone.forward(torch.zeros(1, 3, 256, 256))])
        self.reg_max = 16
        self.no = num_classes + self.reg_max * 4
        self.num_classes = num_classes
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], num_classes)
        self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, num_classes, 1)) for x in ch)
        if not pretrained:
            weights_init(self)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def fuse(self):
        print('Fusing layers... ')
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.forward_fuse
        return self

    def forward(self, input):
        if self.training:
            input, clear_x = input.split((16, 16), dim=0)
        out_features = self.backbone.forward(input)
        [feat1, feat2, feat3] = [out_features[f] for f in range(3)]
        P5_upsample = self.upsample(feat3)
        P4 = torch.cat([P5_upsample, feat2], 1)
        P4 = self.conv3_for_upsample1(P4)
        P4_upsample = self.upsample(P4)
        P3 = torch.cat([P4_upsample, feat1], 1)
        P3 = self.conv3_for_upsample2(P3)
        P3 = self.i1(P3)
        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample, P4], 1)
        P4 = self.conv3_for_downsample1(P4)
        P4 = self.i2(P4)
        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample, feat3], 1)
        P5 = self.conv3_for_downsample2(P5)
        P5 = self.i3(P5)
        dehazing = self.decoder(feat3, feat2, feat1)
        shape = P3.shape
        x = [P3, P4, P5]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.num_classes), 1)
        dbox = self.dfl(box)
        return dbox, cls, x, self.anchors.to(dbox.device), self.strides.to(dbox.device), dehazing