import segmentation_models_pytorch as smp
import ever as er
from module.loss import SegmentationLoss
import torch

@er.registry.MODEL.register()
class AnyUNet(er.ERModule, ):
    def __init__(self, config):
        super(AnyUNet, self).__init__(config)
        self.loss = SegmentationLoss(self.config.loss)
        self.features = smp.Unet(self.config.encoder_name,
                                 encoder_weights=self.config.encoder_weights,
                                 classes=self.config.classes,
                                 activation=None
                                 )

    def forward(self, x, y=None):
        logit = self.features(x)

        if self.training:
            
            return self.loss(logit, y['cls'])

        return logit.softmax(dim=1)

    def set_default_config(self):
        self.config.update(dict(
            encoder_name='resnet50',
            classes=5,
            encoder_weights=None,
            loss=dict(
                ce=dict()
            )
        ))

from segmentation_models_pytorch.unet.model import UnetDecoder, SegmentationHead
from module.baseline.swin.swin_transformer import SwinTransformer
@er.registry.MODEL.register()
class SwinUNet(er.ERModule, ):
    def __init__(self, config):
        super(SwinUNet, self).__init__(config)
        self.loss = SegmentationLoss(self.config.loss)
        self.encoder = SwinTransformer(**self.config.swin)

        self.decoder = UnetDecoder(
            encoder_channels=self.config.out_channels,
            decoder_channels=self.config.decoder_channels,
            n_blocks=4,
            use_batchnorm=True,
            center=False,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.config.decoder_channels[-2],
            out_channels=self.config.classes,
            kernel_size=3,
            upsampling=4
        )

    def forward(self, x, y=None):
        feat_list = self.encoder(x)
        feat = self.decoder(*feat_list)
        logit = self.segmentation_head(feat)
        if self.training:
            return self.loss(logit, y['cls'])

        return logit.softmax(dim=1)

    def set_default_config(self):
        self.config.update(dict(
            swin=dict(
                pretrain_img_size=224,
                embed_dims=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                window_size=7,
                use_abs_pos_embed=False,
                drop_path_rate=0.3,
                patch_norm=True,
                patch_size=4,
                mlp_ratio=4,
                strides=(4, 2, 2, 2),
                out_indices=(0, 1, 2, 3),
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                act_cfg=dict(type='GELU'),
                norm_cfg=dict(type='LN'),
                with_cp=False,
                pretrained=None,
                frozen_stages=-1,
                init_cfg=None
            ),
            out_channels=(128, 256, 512, 1024),
            decoder_channels=(512, 256, 128, 64),
            classes=1,
            loss=dict(
                ce=dict()
            )
        ))

@er.registry.MODEL.register()
class UNetPP(er.ERModule, ):
    def __init__(self, config):
        super(UNetPP, self).__init__(config)
        self.loss = SegmentationLoss(self.config.loss)
        self.features = smp.UnetPlusPlus(self.config.encoder_name,
                                 encoder_weights=self.config.encoder_weights,
                                 classes=self.config.classes,
                                 activation=None
                                 )

    def forward(self, x, y=None):
        logit = self.features(x)

        if self.training:
            return self.loss(logit, y['cls'])

        return logit.softmax(dim=1)

    def set_default_config(self):
        self.config.update(dict(
            encoder_name='resnet50',
            classes=1,
            encoder_weights=None,
            loss=dict(
                ce=dict()
            )
        ))


@er.registry.MODEL.register()
class LinkNet(er.ERModule, ):
    def __init__(self, config):
        super(LinkNet, self).__init__(config)
        self.loss = SegmentationLoss(self.config.loss)
        self.features = smp.Linknet(self.config.encoder_name,
                                 encoder_weights=self.config.encoder_weights,
                                 classes=self.config.classes,
                                 activation=None
                                 )

    def forward(self, x, y=None):
        logit = self.features(x)

        if self.training:
            return self.loss(logit, y['cls'])

        return logit.softmax(dim=1)

    def set_default_config(self):
        self.config.update(dict(
            encoder_name='resnet50',
            classes=1,
            encoder_weights=None,
            loss=dict(
                ce=dict()
            )
        ))



@er.registry.MODEL.register()
class DeepLabV3(er.ERModule, ):
    def __init__(self, config):
        super(DeepLabV3, self).__init__(config)
        self.loss = SegmentationLoss(self.config.loss)
        self.features = smp.DeepLabV3(self.config.encoder_name,
                                    encoder_weights=self.config.encoder_weights,
                                    classes=self.config.classes,
                                    activation=None
                                    )

    def forward(self, x, y=None):
        logit = self.features(x)

        if self.training:
            return self.loss(logit, y['cls'])

        return logit.softmax(dim=1)

    def set_default_config(self):
        self.config.update(dict(
            encoder_name='resnet50',
            classes=1,
            encoder_weights=None,
            loss=dict(
                ce=dict()
            )
        ))


@er.registry.MODEL.register()
class DeepLabV3Plus(er.ERModule, ):
    def __init__(self, config):
        super(DeepLabV3Plus, self).__init__(config)
        self.loss = SegmentationLoss(self.config.loss)
        self.features = smp.DeepLabV3Plus(self.config.encoder_name,
                                      encoder_weights=self.config.encoder_weights,
                                      classes=self.config.classes,
                                      activation=None
                                      )

    def forward(self, x, y=None):
        logit = self.features(x)

        if self.training:
            return self.loss(logit, y['cls'])

        return logit.softmax(dim=1)

    def set_default_config(self):
        self.config.update(dict(
            encoder_name='resnet50',
            classes=1,
            encoder_weights=None,
            loss=dict(
                ce=dict()
            )
        ))


from segmentation_models_pytorch.deeplabv3.model import DeepLabV3PlusDecoder
import torch.nn as nn
@er.registry.MODEL.register()
class SwinDeepLabV3Plus(er.ERModule, ):
    def __init__(self, config):
        super(SwinDeepLabV3Plus, self).__init__(config)
        self.loss = SegmentationLoss(self.config.loss)
        self.encoder = SwinTransformer(**self.config.swin)
        self.decoder = DeepLabV3PlusDecoder(
            self.config.encoder_out_channels,
            out_channels=self.config.decoder_channels,
            atrous_rates=(12, 24, 36),
            output_stride=16,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.config.decoder_channels,
            out_channels=self.config.classes,
            kernel_size=3,
            upsampling=4
        )

    def forward(self, x, y=None):
        feat_list = self.encoder(x)
        feat_list[-1] = nn.UpsamplingBilinear2d(scale_factor=2)(feat_list[-1])
        feat = self.decoder(*feat_list)
        logit = self.segmentation_head(feat)
        if self.training:
            return self.loss(logit, y['cls'])

        return logit.softmax(dim=1)

    def set_default_config(self):
        self.config.update(dict(
            swin=dict(
                pretrain_img_size=224,
                embed_dims=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                window_size=7,
                use_abs_pos_embed=False,
                drop_path_rate=0.3,
                patch_norm=True,
                patch_size=4,
                mlp_ratio=4,
                strides=(4, 2, 2, 2),
                out_indices=(0, 1, 2, 3),
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                act_cfg=dict(type='GELU'),
                norm_cfg=dict(type='LN'),
                with_cp=False,
                pretrained=None,
                frozen_stages=-1,
                init_cfg=None
            ),
            encoder_out_channels=(128, 256, 512, 1024),
            decoder_channels=256,
            classes=1,
            loss=dict(
                ce=dict()
            )
        ))

@er.registry.MODEL.register()
class MANet(er.ERModule, ):
    def __init__(self, config):
        super(MANet, self).__init__(config)
        self.loss = SegmentationLoss(self.config.loss)
        self.features = smp.MAnet(self.config.encoder_name,
                                   encoder_weights=self.config.encoder_weights,
                                   classes=self.config.classes,
                                   activation=None
                                   )

    def forward(self, x, y=None):
        logit = self.features(x)

        if self.training:
            return self.loss(logit, y['cls'])

        return logit.softmax(dim=1)

    def set_default_config(self):
        self.config.update(dict(
            encoder_name='resnet50',
            classes=1,
            encoder_weights=None,
            loss=dict(
                ce=dict()
            )
        ))

@er.registry.MODEL.register()
class PAN(er.ERModule, ):
    def __init__(self, config):
        super(PAN, self).__init__(config)
        self.loss = SegmentationLoss(self.config.loss)
        self.features = smp.PAN(self.config.encoder_name,
                                  encoder_weights=self.config.encoder_weights,
                                  classes=self.config.classes,
                                  activation=None
                                  )

    def forward(self, x, y=None):
        logit = self.features(x)

        if self.training:
            return self.loss(logit, y['cls'])

        return logit.softmax(dim=1)

    def set_default_config(self):
        self.config.update(dict(
            encoder_name='resnet50',
            classes=1,
            encoder_weights=None,
            loss=dict(
                ce=dict()
            )
        ))

