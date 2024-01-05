import torch
from ever.core.checkpoint import load_model_state_dict_from_ckpt
import torch.nn.functional as F
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('ckpt_path', type=str, default=r'./model-30000.pth')

args = parser.parse_args()


def fast_normalize(x):
    x = F.relu(x)
    return x / (torch.sum(x, dim=0, keepdim=True) + 0.0001)

if __name__ == '__main__':
    ckpt_path = args.ckpt_path
    keep_nodes = [1, 2, 3, 4]
    strides = 4
    for kn in keep_nodes:
        ckpt_dir = os.path.dirname(ckpt_path)
        model_state_dict = load_model_state_dict_from_ckpt(ckpt_path)
        cell_argmap_list = []
        con_argmap_list = []
        for k, v in model_state_dict.items():
            if 'weights' in k:
                print(k)
                if 'ops' in k:
                    normalized_v = fast_normalize(v).cpu().numpy()
                    argidx = np.argsort(-normalized_v, kind='stable')
                    cell_argmap_list.append(list(argidx[:kn]))
                else:
                    normalized_v = fast_normalize(v).cpu().numpy()
                    argidx = np.argsort(-normalized_v, kind='stable')
                    con_argmap = np.zeros_like(argidx)
                    con_argmap[argidx[:kn]] = 1
                    con_argmap_list.append(list(con_argmap))
        print('Keep connections: %d, operations: %d' % (kn, kn))

        filtered_map = [len(con_argmap) * [0] for con_argmap in con_argmap_list]
        filtered_map[-strides:] = con_argmap_list[-strides:]
        for i in range(len(con_argmap_list)-1, strides-1, -1):
            for pre_index, used in enumerate(con_argmap_list[i]):
                if pre_index >= 4: # we skip the first four inputs
                    if used == 1:
                        filtered_map[pre_index-4] = con_argmap_list[pre_index-4]

        print(filtered_map)
        print(cell_argmap_list)
        print(len(cell_argmap_list))


        filtered_maparr = np.array(filtered_map)
        np.save(os.path.join(ckpt_dir, 'connection_%d.npy' % kn), filtered_maparr)
        cell_argmaparr = np.array(cell_argmap_list)
        np.save(os.path.join(ckpt_dir, 'operation_%d.npy' % kn), cell_argmaparr)