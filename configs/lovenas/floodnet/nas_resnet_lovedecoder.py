import torch.nn as nn
from configs.base.floodnet import train, test, data, optimizer, learning_rate
from module.nas.nasdecoder import parallel_block


config = dict(
    model=dict(
        type='NasNet',
        params=dict(
            encoder=dict(
                name='resnet50',
                in_channels=3,
                depth=5,
                weights='imagenet'
            ),
            include_os2=False,
            nas_decoder=dict(
                in_strides=(4, 8, 16, 32),
                in_channels=(256, 512, 1024, 2048),
                channels=256,
                stacked_nums=2,
                normalized_fusion='fast_normalize',
                cell_fn=parallel_block
            ),
            head=dict(
                in_channels=256,
                out_channels=128,
                in_feat_output_strides=(4, 8, 16, 32),
                out_feat_output_stride=4,
                norm_fn=nn.BatchNorm2d,
                num_groups_gn=None
            ),
            
            classes=10,
            loss=dict(
                ce=dict()
            )
        )
    ),
    data=data,
    optimizer=optimizer,
    learning_rate=learning_rate,
    train=train,
    test=test
)
