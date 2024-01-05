from albumentations import Compose, OneOf, Normalize
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, RandomCrop, RandomBrightnessContrast
from data.transforms import DivisiblePad
import ever as er
from ever.api.preprocess.albu import RandomDiscreteScale

data = dict(
    train=dict(
        type='FloodNetLoader',
        params=dict(
            image_dir='./FloodNet/train/image_p1024s1024',
            mask_dir='./FloodNet/train/label_p1024s1024',
            transforms=Compose([
                RandomDiscreteScale([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]),
                RandomCrop(512, 512),
                RandomBrightnessContrast(p=0.5),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                RandomRotate90(p=0.5),
                Normalize(mean=(123.675, 116.28, 103.53),
                          std=(58.395, 57.12, 57.375),
                          max_pixel_value=1, always_apply=True),
                er.preprocess.albu.ToTensor()

            ]),
            CV=dict(k=10, i=-1),
            training=True,
            batch_size=16,
            num_workers=4,
        ),
    ),
    test=dict(
        type='FloodNetLoader',
        params=dict(
            image_dir='./FloodNet/test/test-org-img',
            mask_dir='./FloodNet/test/test-label-img',
            transforms=Compose([
                DivisiblePad(size_divisor=32, always_apply=True),
                Normalize(mean=(123.675, 116.28, 103.53),
                          std=(58.395, 57.12, 57.375),
                          max_pixel_value=1, always_apply=True),
                er.preprocess.albu.ToTensor()
            ]),
            CV=dict(k=10, i=-1),
            training=False,
            batch_size=1,
            num_workers=0,
        ),
    ),
)
optimizer = dict(
    type='sgd',
    params=dict(
        momentum=0.9,
        weight_decay=0.0001
    ),
    grad_clip=dict(
        max_norm=35,
        norm_type=2,
    )
)
learning_rate = dict(
    type='poly',
    params=dict(
        base_lr=0.01,
        power=0.9,
        max_iters=60000,
    ))
train = dict(
    forward_times=1,
    num_iters=60000,
    eval_per_epoch=False,
    summary_grads=False,
    summary_weights=False,
    distributed=True,
    apex_sync_bn=True,
    sync_bn=True,
    eval_after_train=True,
    log_interval_step=50,
    save_ckpt_interval_epoch=50,
    eval_interval_epoch=50,
)

test = dict(

)
