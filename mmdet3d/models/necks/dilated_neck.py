# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np
from mmcv.cnn import ConvModule, build_conv_layer
from mmcv.runner import BaseModule
from torch import nn as nn
import torch.nn.functional as F
from mmdet.models.builder import NECKS
import torch

from mmdet3d.models.necks.dla_neck import fill_up_weights


class Conv(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1, act='relu'):
        super(Conv, self).__init__()
        if act is not None:
            if act == 'relu':
                self.convs = nn.Sequential(
                    nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=False),
                    nn.BatchNorm2d(c2),
                    nn.ReLU(inplace=True) if act else nn.Identity()
                )
            elif act == 'leaky':
                self.convs = nn.Sequential(
                    nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=False),
                    nn.BatchNorm2d(c2),
                    nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()
                )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=False),
                nn.BatchNorm2d(c2)
            )

    def forward(self, x):
        return self.convs(x)


class UpSample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corner=None):
        super(UpSample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corner = align_corner

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor,
                                                mode=self.mode, align_corners=self.align_corner)


class ResizeConv(nn.Module):
    def __init__(self, c1, c2, act='relu', size=None, scale_factor=None, mode='nearest', align_corner=None):
        super(ResizeConv, self).__init__()
        self.upsample = UpSample(size=size, scale_factor=scale_factor, mode=mode, align_corner=align_corner)
        self.conv = Conv(c1, c2, k=1, act=act)

    def forward(self, x):
        x = self.conv(self.upsample(x))
        return x


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c, d=1, e=0.5, act='relu'):
        super(Bottleneck, self).__init__()
        c_ = int(c * e)
        self.branch = nn.Sequential(
            Conv(c, c_, k=1, act=act),
            Conv(c_, c_, k=3, p=d, d=d, act=act),
            Conv(c_, c, k=1, act=act)
        )

    def forward(self, x):
        return x + self.branch(x)


class DilateEncoder(nn.Module):
    """ DilateEncoder """
    def __init__(self, c1, c2, act='relu', dilation_list=[2, 4, 6, 8]):
        super(DilateEncoder, self).__init__()
        self.projector = nn.Sequential(
            Conv(c1, c2, k=1, act=None),
            Conv(c2, c2, k=3, p=1, act=None)
        )
        encoders = []
        for d in dilation_list:
            encoders.append(Bottleneck(c=c2, d=d, act=act))
        self.encoders = nn.Sequential(*encoders)

    def forward(self, x):
        x = self.projector(x)
        x = self.encoders(x)

        return x


class SPP(nn.Module):
    """
        Spatial Pyramid Pooling
    """
    def __init__(self, c1, c2, e=0.5, act='relu'):
        super(SPP, self).__init__()
        c_ = int(c1 * e)
        self.cv1 = Conv(c1, c_, k=1, act=act)
        self.cv2 = Conv(c_*4, c2, k=1, act=act)

    def forward(self, x):
        x = self.cv1(x)
        x_1 = F.max_pool2d(x, 5, stride=1, padding=2)
        x_2 = F.max_pool2d(x, 9, stride=1, padding=4)
        x_3 = F.max_pool2d(x, 13, stride=1, padding=6)
        x = torch.cat([x, x_1, x_2, x_3], dim=1)
        x = self.cv2(x)

        return x

@NECKS.register_module()
class DilatedNeck(BaseModule):
    """DilatedNeck Neck.

    Args:
        in_channels (list[int], optional): List of input channels
            of multi-scale feature map.
        start_level (int, optional): The scale level where upsampling
            starts. Default: 2.
        end_level (int, optional): The scale level where upsampling
            ends. Default: 5.
        norm_cfg (dict, optional): Config dict for normalization
            layer. Default: None.
        use_dcn (bool, optional): Whether to use dcn in IDAup module.
            Default: True.
    """

    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 out_channels=[256, 256, 256, 256],
                 norm_cfg=None,
                 init_cfg=None):
        super(DilatedNeck, self).__init__(init_cfg)
        c2, c3, c4, c5 = in_channels
        p2, p3, p4, p5 = out_channels
        act='relu'
        self.neck = DilateEncoder(c1=c5, c2=p5, act=act)
        # upsample
        self.deconv4 = ResizeConv(c1=p5, c2=p4, act=act, scale_factor=2)  # 32 -> 16
        self.latter4 = Conv(c4, p4, k=1, act=None)
        self.smooth4 = Conv(p4, p4, k=3, p=1, act=act)

        self.deconv3 = ResizeConv(c1=p4, c2=p3, act=act, scale_factor=2)  # 16 -> 8
        self.latter3 = Conv(c3, p3, k=1, act=None)
        self.smooth3 = Conv(p3, p3, k=3, p=1, act=act)

        # self.deconv2 = ResizeConv(c1=p3, c2=p2, act=act, scale_factor=2)  # 8 -> 4
        # self.latter2 = Conv(c2, p2, k=1, act=None)
        # self.smooth2 = Conv(p2, p2, k=3, p=1, act=act)

    def forward(self, x):
        # mlvl_features = [x[i] for i in range(len(x))]
        c2,c3,c4,c5 = x
        p5 = self.neck(c5)
        p4 = self.smooth4(self.latter4(c4) + self.deconv4(p5))
        p3 = self.smooth3(self.latter3(c3) + self.deconv3(p4))
        # p2 = self.smooth2(self.latter2(c2) + self.deconv2(p3))

        return (p3,p4,p5)
    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                # In order to be consistent with the source code,
                # reset the ConvTranspose2d initialization parameters
                m.reset_parameters()
                # Simulated bilinear upsampling kernel
                fill_up_weights(m)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
                m.reset_parameters()
