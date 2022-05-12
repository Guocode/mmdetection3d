import torch
from mmcv.cnn import ConvModule
import torch.nn as nn


class ConvBEVTransformer(nn.Module):
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
        bev_convs.append(
            ConvModule(self.feat_channels, self.out_channels, (1, 1), (1, 1), (0, 0),norm_cfg=self.norm_cfg)
        )
        self.bev_convs = nn.Sequential(*bev_convs)


if __name__ == '__main__':
    layer = ConvBEVTransformer()
    x = torch.rand((1, 128, 128, 128), dtype=torch.float32)
    bev_feats = layer(x)
    print(bev_feats.shape)
