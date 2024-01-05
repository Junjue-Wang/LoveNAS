import torch.nn as nn
from ever.module.fpn import FastNormalizedFusionConv3x3, NormalizedFusionConv3x3
import torch
import numpy as np
from module.nas.operations import ParallelCell

def parallel_block(in_channels, out_channels, ops_list=[0, 1, 2, 3], fusion_op='search', scale_factor=1.0):
    return nn.Sequential(ParallelCell(in_channels, out_channels, ops_list=ops_list, fusion_op=fusion_op),
                         nn.UpsamplingBilinear2d(scale_factor=scale_factor) if scale_factor !=1.0 else
                         nn.Identity())


class NasDecoder(nn.Module):
    def __init__(self, in_strides:tuple, in_channels, channels=256,
                 stacked_nums=2, normalized_fusion='normalize', cell_fn=parallel_block):
        super(NasDecoder, self).__init__()
        self.in_strides = in_strides
        self.channels = channels
        self.cell_fn = cell_fn
        self.in_channels = in_channels
        nf_op = FastNormalizedFusionConv3x3 if normalized_fusion == 'fast_normalize' else NormalizedFusionConv3x3
        self.fusion_modules = nn.ModuleList()
        # init fusion
        for i in range(stacked_nums):
            for c in range(len(in_channels)):
                if i == 0:
                    self.fusion_modules.append(nf_op(len(self.in_strides), channels, channels))
                else:
                    self.fusion_modules.append(nf_op((i+1)*len(self.in_strides), channels, channels))
        # init ops
        self.ops = nn.ModuleList()
        for i in range(stacked_nums):
            for s in self.in_strides:
                init_ops = self.generate_ops(s, init=True)
                for _ in range(i):
                    init_ops += self.generate_ops(s)
                self.ops.append(init_ops)

    def generate_ops(self, out_s, init=False):
        ops = nn.ModuleList()
        if init is False:
            for in_s in self.in_strides:
                ops.append(self.cell_fn(self.channels, self.channels, scale_factor= in_s / out_s))
        else:
            for in_s, in_c in zip(self.in_strides, self.in_channels):
                ops.append(self.cell_fn(in_c, self.channels, scale_factor=in_s / out_s))
        return ops

    def forward(self, features_list):
        inter_features = features_list
        temp_features = []
        for ops, nf_op in zip(self.ops, self.fusion_modules):
            # get multiple feats
            feats = [op(feat) for op, feat in zip(ops, inter_features)]
            # fusion
            temp_features.append(nf_op(feats))
            if len(temp_features) % len(self.in_strides) == 0:
                inter_features += temp_features
                temp_features = []

        return inter_features[-len(self.in_strides):]

    def arch_params(self):
        for fusion_module in self.fusion_modules:
            print(fusion_module._modules['0'].weights)
        # if has cell search
        if self.cell_fn is parallel_block:
            for cell_module in self.ops:
                print(cell_module._modules['0'].weights)


class ParseDecoder(nn.Module):
    def __init__(self, connect_map, ops_map, in_strides:tuple, in_channels, channels=256, cell_fn=parallel_block, fusion_op='cat'):
        super(ParseDecoder, self).__init__()
        if isinstance(connect_map, list):
            self.connect_map = connect_map
        else:
            self.connect_map = np.load(connect_map, allow_pickle=True).tolist()
            print('Load connections', self.connect_map)

        if isinstance(ops_map, list):
            self.ops_map = ops_map
        elif isinstance(ops_map, str):
            self.ops_map = np.load(ops_map, allow_pickle=True).tolist()
            print('Load operations', self.ops_map)
        else:
            print('No searched operations, default', ops_map)

        self._check_map()
        self.in_strides = in_strides
        self.channels = channels
        self.in_channels = in_channels
        self.fusion_modules = nn.ModuleList()
        # init ops
        self.ops = nn.ModuleList()
        op_dix = 0
        for out_s_i, con_map in enumerate(self.connect_map):
            node_ops = nn.ModuleList()
            out_s = self.in_strides[(out_s_i % len(self.in_strides))]
            for in_s_i, node in enumerate(con_map):
                in_s = self.in_strides[(in_s_i % len(self.in_strides))]
                # select existing node
                if node == 1:
                    if in_s_i < len(self.in_strides):
                        if ops_map is None:
                            node_ops.append(cell_fn(self.in_channels[in_s_i], self.channels, scale_factor=in_s / out_s))
                        else:
                            node_ops.append(cell_fn(self.in_channels[in_s_i], self.channels,
                                                    ops_list=self.ops_map[op_dix], fusion_op=fusion_op, scale_factor=in_s / out_s))
                    else:
                        if ops_map is None:
                            node_ops.append(cell_fn(self.channels, self.channels, scale_factor=in_s / out_s))
                        else:
                            node_ops.append(cell_fn(self.channels, self.channels,
                                                    ops_list=self.ops_map[op_dix], fusion_op=fusion_op, scale_factor=in_s / out_s))
                op_dix += 1
            self.ops.append(node_ops)


    def forward(self, features_list):
        temp_features = []
        for idx, (ops, con_map) in enumerate(zip(self.ops, self.connect_map)):
            # get multiple feats
            if sum(con_map) > 0:
                node_inputs = [features_list[idx] for idx, con in enumerate(con_map) if con == 1]
                feats = [op(feat) for op, feat in zip(ops, node_inputs)]
                # mean fusion
                fusion_result = sum(feats) / len(feats)
                temp_features.append(fusion_result)
            else:
                temp_features.append(None)
            if len(temp_features) % len(self.in_strides) == 0:
                features_list += temp_features
                temp_features = []
        
        output_features = features_list[-len(self.in_strides):]
        
        return output_features

    def _check_map(self):
        check_map = [True, True, True, True]
        for node in self.connect_map:
            if sum(node) > 0:
                for in_idx, status in enumerate(node):
                    if status > 0:
                        if check_map[in_idx] is False:
                            raise ValueError('Please check connect map!')
                check_map.append(True)
            else:
                check_map.append(False)