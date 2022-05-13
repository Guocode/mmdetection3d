import torch
from mmcv.cnn import ConvModule
import torch.nn as nn
import torch.nn.functional as F


class HSEBlock(nn.Module):
    def __init__(self, pool_func=F.adaptive_max_pool2d, init_cfg=None):
        super(HSEBlock, self).__init__()
        self.pool_func = pool_func

    def forward(self, x):
        px = self.pool_func(x, (1, x.size(2)))
        x = px+x
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
        bev_feats = self.bev_convs(x)
        return bev_feats

    def init_weights(self):
        pass

    def _init_layers(self):
        bev_convs = []
        for i in range(self.stacked_bev_convs_num - 1):
            chn = self.in_channels if i == 0 else self.feat_channels
            groups = chn if chn == self.feat_channels else 1
            bev_convs.append(
                ConvModule(chn, self.feat_channels, (self.kernel_s, 1), (1, 1), (self.kernel_s // 2, 0), groups=groups,
                           norm_cfg=self.norm_cfg)
            )
            bev_convs.append(HSEBlock())
        bev_convs.append(
            ConvModule(self.feat_channels, self.out_channels, (1, 1), (1, 1), (0, 0), norm_cfg=self.norm_cfg)
        )
        self.bev_convs = nn.Sequential(*bev_convs)


if __name__ == '__main__':
    layer = ConvBEVTransformer(in_channels=64)
    x = torch.rand((1, 64, 128, 128), dtype=torch.float32)
    bev_feats = layer(x)
    print(bev_feats.shape)

    # layer = HSEBlock()
    # x = layer(x)
    # print(x.shape)
