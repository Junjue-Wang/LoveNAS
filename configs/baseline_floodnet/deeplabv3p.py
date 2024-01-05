from configs.base.floodnet import train, test, data, optimizer, learning_rate

config = dict(
    model=dict(
        type='DeepLabV3Plus',
        params=dict(
            encoder_name='resnet50',
            classes=10,
            encoder_weights='imagenet',
            loss=dict(
                ce=dict()
            )
        )
    ),
    data=data,
    optimizer=optimizer,
    learning_rate=learning_rate,
    train=train,
    test=dict(
      tta=False
    )
)
