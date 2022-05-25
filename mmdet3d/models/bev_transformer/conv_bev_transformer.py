import torch
from mmcv.cnn import ConvModule
import torch.nn as nn
import torch.nn.functional as F


class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()
        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)
        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1
        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)
        if self.with_r:
            rr = torch.sqrt(
                torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5,
                                                                                 2))
            ret = torch.cat([ret, rr], dim=1)
        return ret


class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret


class HSEBlock(nn.Module):
    def __init__(self, inchn, outchn, pool_func=F.adaptive_max_pool2d, init_cfg=None):
        super(HSEBlock, self).__init__()
        self.hpool_func = pool_func
        self.vpool_func = pool_func
        self.inchn = inchn
        self.outchn = outchn
        self.conv_h = ConvModule(outchn, outchn, (9, 1), (1, 1), (4, 0), groups=outchn,
                                 norm_cfg=dict(type='BN'))
        self.conv_v = ConvModule(outchn, outchn, (1, 9), (1, 1), (0, 4), groups=outchn,
                                 norm_cfg=dict(type='BN'))
        self.conv_n = ConvModule(outchn, outchn, (3, 3), (1, 1), (1, 1), groups=outchn,
                                 norm_cfg=dict(type='BN'))
        if inchn != outchn:
            self.res_conv = ConvModule(inchn, outchn, (1, 1), (1, 1), (0, 0), groups=1,
                                       norm_cfg=dict(type='BN'))

    def forward(self, x):
        hpx = self.hpool_func(x, (1, x.size(3)))
        vpx = self.vpool_func(x, (x.size(2), 1))

        if self.inchn != self.outchn:
            x = self.res_conv(x * torch.sigmoid(hpx + vpx))
        else:
            x = x * torch.sigmoid(hpx + vpx)

        cx = self.conv_n(self.conv_v(self.conv_h(x)))
        return cx + x
        # if self.inchn!=self.outchn:
        #     x = self.res_conv(x + px) + cx
        # else:
        #     x = x + px + cx
        # return x


class ConvBEVTransformer(nn.Module):
    def __init__(self,
                 in_channels=128,
                 out_channels=128,
                 in_h=12,
                 z_size=32,
                 h_pool=1,
                 z_up=8,
                 feat_channels=128,
                 stacked_bev_convs_num=3,
                 kernel_s=9,
                 norm_cfg=dict(type='BN'),
                 init_cfg=None):
        super(ConvBEVTransformer, self).__init__()
        self.in_channels = in_channels
        self.in_h = in_h
        self.h_pool = h_pool
        self.z_up = z_up
        self.z_size = z_size
        self.out_channels = out_channels
        self.feat_channels = feat_channels
        self.stacked_bev_convs_num = stacked_bev_convs_num
        self.norm_cfg = norm_cfg
        self.kernel_s = kernel_s

        self._init_layers()

    def forward(self, x):
        # x = self.pre_coordconv(x)
        x = F.max_pool2d(x, (self.h_pool, 1))
        n, c, h, w = x.shape
        x = x.reshape(n, c * h, 1, w)
        x = self.img2bev_conv(x)
        x = x.repeat(1, 1, self.z_size//self.z_up, 1)
        # x = F.upsample(x,(self.z_size,w))
        for bev_conv in self.bev_convs:
            x = F.upsample(x, scale_factor=(2, 1))
            x = bev_conv(x)+x
        # bev_feats = self.bev_convs_l(bev_feats)
        # bev_feats = self.coordconv(bev_feats)
        return x

    def init_weights(self):
        pass

    def _init_layers(self):
        # self.pre_conv = ConvModule(self.in_channels * self.in_h // self.h_pool, self.feat_channels, 1,
        #                            norm_cfg=self.norm_cfg)
        # self.pre_coordconv = CoordConv(self.in_channels, self.feat_channels, kernel_size=(3, 3), padding=(1, 1))
        self.img2bev_conv = nn.Sequential(
            ConvModule(self.in_channels*self.in_h//self.h_pool, self.feat_channels, 1,
                       norm_cfg=self.norm_cfg),
            # ConvModule(self.feat_channels , self.feat_channels* self.z_size//self.z_up, 1,
            #            norm_cfg=self.norm_cfg)
        )

        # self.coordconv = CoordConv(self.feat_channels, self.feat_channels, kernel_size=(3, 3), padding=(1, 1))
        bev_convs = []
        for i in range(self.stacked_bev_convs_num):
            chn = self.feat_channels
            bev_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg)
            )
        self.bev_convs = nn.Sequential(*bev_convs)


if __name__ == '__main__':
    layer = ConvBEVTransformer(in_channels=64)
    x = torch.rand((1, 64, 12, 40), dtype=torch.float32)
    bev_feats = layer(x)
    print(bev_feats.shape)

    # layer = HSEBlock()
    # x = layer(x)
    # print(x.shape)
