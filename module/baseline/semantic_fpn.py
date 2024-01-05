import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ever.interface import ERModule
from ever import registry
from module.baseline.base import AssymetricDecoder, FPN, default_conv_block
import math
from module.loss import SegmentationLoss
from segmentation_models_pytorch.encoders import get_encoder
from module.baseline.swin.swin_transformer import SwinTransformer

@registry.MODEL.register('SemanticFPN')
class SemanticFPN(ERModule):
    def __init__(self, config):
        super(SemanticFPN, self).__init__(config)
        if 'swin' in self.config.encoder:
            self.en = SwinTransformer(**self.config.encoder.swin)
        else:
            self.en = get_encoder(**self.config.encoder)
        self.fpn = FPN(**self.config.fpn)
        self.decoder = AssymetricDecoder(**self.config.decoder)
        self.cls_pred_conv = nn.Conv2d(self.config.decoder.out_channels, self.config.classes, 1)
        self.upsample4x_op = nn.UpsamplingBilinear2d(scale_factor=4)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.cls_loss = SegmentationLoss(self.config.loss)


    def forward(self, x, y=None):
        if 'swin' in self.config.encoder:
            feat_list = self.en(x)
        else:
            feat_list = self.en(x)[1:]
        fpn_feat_list = self.fpn(feat_list)
        final_feat = self.decoder(fpn_feat_list)
        cls_pred = self.cls_pred_conv(final_feat)
        cls_pred = self.upsample4x_op(cls_pred)
        if self.training:
            cls_true = y['cls']
            #loss_dict = dict()
            loss_dict = self.cls_loss(cls_pred, cls_true)
            mem = torch.cuda.max_memory_allocated() // 1024 // 1024
            loss_dict['mem'] = torch.from_numpy(np.array([mem], dtype=np.float32)).to(self.device)
            return loss_dict
        else:
            if 'mbloss' in self.config.loss:
                cls_prob = torch.sigmoid(cls_pred)
            else:
                cls_prob = torch.softmax(cls_pred, dim=1)
            return cls_prob



    def set_default_config(self):
        self.config.update(dict(
            encoder=dict(
                name='resnet50',
                weights='imagenet',
                in_channels=3
            ),
            fpn=dict(
                in_channels_list=(256, 512, 1024, 2048),
                out_channels=256,
                conv_block=default_conv_block,
                top_blocks=None,
            ),
            decoder=dict(
                in_channels=256,
                out_channels=128,
                in_feat_output_strides=(4, 8, 16, 32),
                out_feat_output_stride=4,
                norm_fn=nn.BatchNorm2d,
                num_groups_gn=None
            ),
            classes=7,
            loss=dict(
                ignore_index=255,
            )
        ))


