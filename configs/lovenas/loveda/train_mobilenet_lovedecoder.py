import torch.nn as nn
from configs.base.loveda import train, test, data, optimizer, learning_rate
from module.nas.nasdecoder import parallel_block
config = dict(
    model=dict(
        type='NasNet',
        params=dict(
            encoder=dict(
                name='mobilenet_v2',
                in_channels=3,
                depth=5,
                weights='imagenet',
                nas_weights='./logs/loveda/mobilenet_lovedecoder/mobilenet_loveda_30k.pth',
            ),
            include_os2=False,
            parse_decoder=dict(
                in_strides=(4, 8, 16, 32),
                in_channels=(24, 32, 96, 1280),
                channels=128,
                connect_map='./searched_archs/mobilenet_lovedecoder/c3.npy',
                ops_map='./searched_archs/mobilenet_lovedecoder/p2.npy',
                cell_fn=parallel_block
            ),
            head=dict(
                in_channels=128,
                out_channels=64,
                in_feat_output_strides=(4, 8, 16, 32),
                out_feat_output_stride=4,
                norm_fn=nn.BatchNorm2d,
                num_groups_gn=None
            ),
            classes=7,
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
