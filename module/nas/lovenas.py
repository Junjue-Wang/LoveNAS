import torch
import torch.nn as nn
import numpy as np
from ever.interface import ERModule
from ever import registry
from module.loss import SegmentationLoss
from module.nas.nasdecoder import NasDecoder, ParseDecoder
import math
from segmentation_models_pytorch.encoders import get_encoder
from ever.core.checkpoint import load_model_state_dict_from_ckpt
from module.baseline.swin.swin_transformer import SwinTransformer
import torch.nn.functional as F

class FusionHead(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_feat_output_strides=(4, 8, 16, 32),
                 out_feat_output_stride=4,
                 norm_fn=nn.BatchNorm2d,
                 num_groups_gn=None):
        super(FusionHead, self).__init__()
        if norm_fn == nn.BatchNorm2d:
            norm_fn_args = dict(num_features=out_channels)
        elif norm_fn == nn.GroupNorm:
            if num_groups_gn is None:
                raise ValueError('When norm_fn is nn.GroupNorm, num_groups_gn is needed.')
            norm_fn_args = dict(num_groups=num_groups_gn, num_channels=out_channels)
        else:
            raise ValueError('Type of {} is not support.'.format(type(norm_fn)))
        self.blocks = nn.ModuleList()
        for in_feat_os in in_feat_output_strides:
            num_upsample = int(math.log2(int(in_feat_os))) - int(math.log2(int(out_feat_output_stride)))

            num_layers = num_upsample if num_upsample != 0 else 1

            self.blocks.append(nn.Sequential(*[
                nn.Sequential(
                    nn.Conv2d(in_channels if idx == 0 else out_channels, out_channels, 3, 1, 1, bias=False),
                    norm_fn(**norm_fn_args) if norm_fn is not None else nn.Identity(),
                    nn.ReLU(inplace=True),
                    nn.UpsamplingBilinear2d(scale_factor=2) if num_upsample != 0 else nn.Identity(),
                )
                for idx in range(num_layers)]))
        self.in_feat_output_strides = in_feat_output_strides

    def forward(self, feat_list: list):
        inner_feat_list = []
        for idx, block in enumerate(self.blocks):
            decoder_feat = block(feat_list[idx])
            inner_feat_list.append(decoder_feat)

        out_feat = sum(inner_feat_list) / len(self.in_feat_output_strides)
        return out_feat



def load_nas_encoder_weights(encoder:nn.Module, weight_path):

    encoder_model_params = load_model_state_dict_from_ckpt(weight_path)
    filtered_encoder_model_params = dict()

    for k, v in encoder_model_params.items():
        if 'en.' in k:
            filtered_encoder_model_params[k.replace('en.', '')] = v

    if '_fc.' in encoder.state_dict().keys():
        filtered_encoder_model_params['_fc.bias'] = 1
        filtered_encoder_model_params['_fc.weight'] = 1
    elif 'fc.' in encoder.state_dict().keys():
        filtered_encoder_model_params['fc.bias'] =1
        filtered_encoder_model_params['fc.weight'] =1
    encoder.load_state_dict(filtered_encoder_model_params, strict=True)
    print('load lovenas weights!')



@registry.MODEL.register()
class NasNet(ERModule):
    def __init__(self, config):
        super(NasNet, self).__init__(config)
       
        if 'swin' in self.config.encoder:
            self.en = SwinTransformer(**self.config.encoder.swin)
        else:
            self.en = get_encoder(self.config.encoder.name, self.config.encoder.in_channels, weights=self.config.encoder.weights)

        if 'nas_weights_path' in self.config.encoder:
            load_nas_encoder_weights(self.en, self.config.encoder.nas_weights_path)

        if 'nas_decoder' in self.config:
            self.decoder = NasDecoder(**self.config.nas_decoder)
        else:
            self.decoder = ParseDecoder(**self.config.parse_decoder)
        self.head = FusionHead(**self.config.head)
        self.cls_pred_conv = nn.Conv2d(self.config.head.out_channels, self.config.classes, 1)
        self.upsample4x_op = nn.UpsamplingBilinear2d(scale_factor=self.config.head.out_feat_output_stride)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.cls_loss = SegmentationLoss(self.config.loss)


    def forward(self, x, y=None):
        if 'swin' in self.config.encoder:
            feat_list = self.en(x)
        else:
            feat_list = self.en(x)[2:]

        multi_scale_feats = self.decoder(feat_list)
        final_feat = self.head(multi_scale_feats)
        cls_pred = self.cls_pred_conv(final_feat)
        cls_pred = self.upsample4x_op(cls_pred)
        if self.training:
            cls_true = y['cls']
            loss_dict = self.cls_loss(cls_pred, cls_true)
            mem = torch.cuda.max_memory_allocated() // 1024 // 1024
            loss_dict['mem'] = torch.from_numpy(np.array([mem], dtype=np.float32)).to(self.device)
            return loss_dict

        cls_prob = torch.softmax(cls_pred, dim=1)

        return cls_prob

    def set_default_config(self):
        self.config.update(dict(
            encoder=dict(
                name='resnet50',
                in_channels=3,
                weights='imagenet'
            ),
            include_os2=False,
            head=dict(
                in_channels=128,
                out_channels=64,
                in_feat_output_strides=(4, 8, 16, 32),
                out_feat_output_stride=4,
                norm_fn=nn.BatchNorm2d,
                num_groups_gn=None
            ),
            parse_decoder=dict(
                connect_map = [[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [1, 0 , 0, 1 ,1 ,0 ,0 ,0], [1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1]],
                in_strides=(4, 8, 16, 32),
                in_channels=(256, 512, 128, 128),
                channels=128,
            ),
            classes=7,
            loss=dict(
                ce=dict()
            )
        ))


if __name__ == '__main__':
    from ever.util.param_util import count_model_parameters
    x = torch.ones(2, 3, 512, 512)
    m = NasNet(dict(
        meta_connection=dict(
            in_chans=[None, None, 1024, 2048],
            patch_sizes=[None, None, 8, 4],
            meta=dict(
                embed_dim=128,
                pool_size=3,
                mlp_ratio=4.,
                act_layer=nn.GELU,
                norm_layer=GroupNorm,
                drop=0.,
                drop_path=0.,
                use_layer_scale=True,
                layer_scale_init_value=1e-5
            ),

        )
    )).eval()
    o = m(x)
    print(o.shape)


