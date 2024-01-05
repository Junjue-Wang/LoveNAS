import torch.nn as nn
from module.nas.nasdecoder import parallel_block
from configs.base.floodnet import train, test, data, optimizer, learning_rate

config = dict(
    model=dict(
        type='NasNet',
        params=dict(
            encoder=dict(
                name='efficientnet-b3',
                in_channels=3,
                depth=5,
                weights='imagenet',
                nas_weights='./logs/floodnet/efnetb3_lovedecoder/efnetb3_floodnet_60k.pth',
            ),
            include_os2=False,
            parse_decoder=dict(
                in_strides=(4, 8, 16, 32),
                in_channels=(32, 48, 136, 384),
                channels=384,
                cell_fn=parallel_block,
                connect_map='./searched_archs/floodnet/efnetb3_lovedecoder/c4.npy',
                ops_map='./searched_archs/floodnet/efnetb3_lovedecoder/p3.npy',
            ),
            head=dict(
                in_channels=384,
                out_channels=128,
                in_feat_output_strides=(4, 8, 16, 32),
                out_feat_output_stride=4,
                norm_fn=nn.BatchNorm2d,
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
