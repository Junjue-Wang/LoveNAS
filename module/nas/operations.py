import torch.nn as nn
from ever.module.fpn import FastNormalizedFusionConv3x3
import torch
from collections import OrderedDict
import torch.nn.functional as F


class SeparableConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True, padding_mode='zeros'):
        super(SeparableConv2d, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                      bias=False, padding_mode=padding_mode),
            nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        )


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=False,
                 bn=True,
                 relu=True,
                 init_fn=None):
        super(ConvBlock, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                      padding, dilation, groups,
                      bias),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            nn.ReLU(True) if relu else nn.Identity()
        )
        if init_fn:
            self.apply(init_fn)

    @staticmethod
    def same_padding(kernel_size, dilation):
        return dilation * (kernel_size - 1) // 2


class PoolBlock(nn.Sequential):
    def __init__(self, output_size, in_channels, out_channels, bn=True, relu=True):
        super(PoolBlock, self).__init__(
            nn.AdaptiveAvgPool2d(output_size),
            SeparableConv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            nn.ReLU(True) if relu else nn.Identity()
        )

    def forward(self, x: torch.Tensor):
        size = x.shape[-2:]
        for m in self:
            x = m(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


Cell_OPS = OrderedDict(
     conv3x3 = lambda channel: nn.Conv2d(channel, channel,3, stride=1, padding= 1, bias=False, groups=channel),
     conv5x5 = lambda channel: nn.Conv2d(channel, channel,5, stride=1, padding= 2, bias=False, groups=channel),
     conv7x7 = lambda channel: nn.Conv2d(channel, channel,7, stride=1, padding= 3, bias=False, groups=channel),
     global_pool = lambda channel: PoolBlock(1, channel, channel),
)


class ParallelCell(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ops_list = [0, 1, 2, 3],
                 fusion_op = 'search'
                 ):
        super(ParallelCell, self).__init__()
        
        self.proj_conv1 = ConvBlock(in_channels, out_channels, 1)
        #assert out_channels % len(ops_list) == 0
        self.interval_channel = out_channels // len(ops_list)
        self.parallel_conv_block = nn.ModuleList(
            [list(Cell_OPS.values())[op_idx](self.interval_channel) for op_idx in ops_list]
        )
        if fusion_op is 'search':
            self.fusion = FastNormalizedFusionConv3x3(len(ops_list), self.interval_channel, out_channels)
        else:
            self.fusion = fusion_op
            self.proj_conv2 = ConvBlock(out_channels, out_channels, 1) if self.fusion == 'cat' else \
                ConvBlock(self.interval_channel, out_channels, 1)


    def forward(self, x):
        x = self.proj_conv1(x)
        out_list = []
        for idx, conv_block in enumerate(self.parallel_conv_block):
            out_i = conv_block(x[:, idx*self.interval_channel:(idx+1)*self.interval_channel, :, :])
            out_list.append(out_i)
        if self.fusion == 'mean':
            return self.proj_conv2(sum(out_list) / len(out_list))
        elif self.fusion == 'cat':
            return self.proj_conv2(torch.cat(out_list, dim=1))
        else:
            return self.fusion(out_list)