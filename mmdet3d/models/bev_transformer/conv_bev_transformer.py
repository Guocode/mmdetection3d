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
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)
        return ret

class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels+2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)
    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret
class HSEBlock(nn.Module):
    def __init__(self, inchn,outchn,pool_func=F.adaptive_max_pool2d, init_cfg=None):
        super(HSEBlock, self).__init__()
        self.pool_func = pool_func
        self.inchn = inchn
        self.outchn = outchn
        self.conv_h= ConvModule(outchn, outchn, (9, 1), (1, 1), (4, 0), groups=outchn,
                           norm_cfg=dict(type='BN'))
        self.conv_n= ConvModule(outchn, outchn, (3, 3), (1, 1), (1, 1), groups=outchn,
                           norm_cfg=dict(type='BN'))
        if inchn!=outchn:
            self.res_conv = ConvModule(inchn, outchn, (1, 1), (1, 1), (0, 0), groups=1,
                           norm_cfg=dict(type='BN'))
    def forward(self, x):
        px = self.pool_func(x, (1, x.size(3)))
        if self.inchn!=self.outchn:
            x = self.res_conv(x + px)
        else:
            x = x+px

        cx = self.conv_n(self.conv_h(x))
        return cx+x
        if self.inchn!=self.outchn:
            x = self.res_conv(x + px) + cx
        else:
            x = x + px + cx
        return x


class ConvBEVTransformer(nn.Module):
    def __init__(self,
                 in_channels=128,
                 out_channels=128,
                 feat_channels=128,
                 stacked_bev_convs_num=5,
                 kernel_s=9,
                 norm_cfg=dict(type='BN'),
                 init_cfg=None):
        super(ConvBEVTransformer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feat_channels = feat_channels
        self.stacked_bev_convs_num = stacked_bev_convs_num
        self.norm_cfg = norm_cfg
        self.kernel_s = kernel_s

        self._init_layers()

    def forward(self, x):
        x = self.coordconv(x)
        bev_feats = self.bev_convs(x)
        # bev_feats = self.bev_convs_l(bev_feats)
        return bev_feats

    def init_weights(self):
        pass

    def _init_layers(self):
        self.coordconv = CoordConv(self.in_channels,self.in_channels,kernel_size=(3,3),padding=(1,1))
        bev_convs = []
        for i in range(self.stacked_bev_convs_num - 1):
            chn = self.in_channels if i == 0 else self.feat_channels
            bev_convs.append(HSEBlock(chn,self.feat_channels))
        self.bev_convs = nn.Sequential(*bev_convs)


if __name__ == '__main__':
    layer = ConvBEVTransformer(in_channels=64)
    x = torch.rand((1, 64, 128, 128), dtype=torch.float32)
    bev_feats = layer(x)
    print(bev_feats.shape)

    # layer = HSEBlock()
    # x = layer(x)
    # print(x.shape)
