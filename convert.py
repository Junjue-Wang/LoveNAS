import ever as er
from ever.core.builder import make_model, make_dataloader
import torch
import numpy as np
import os
from data.loveda import COLOR_MAP
from tqdm import tqdm
from module.tta import tta, Scale, HorizontalFlip, VerticalFlip, Rotate90k
import logging
from ever.core.checkpoint import load_model_state_dict_from_ckpt
from ever.core.config import import_config
from train_loveda import seed_torch
from skimage.io import imsave
import argparse

parser = argparse.ArgumentParser(description='Infer methods')
parser.add_argument('--ckpt_path',  type=str,
                    help='ckpt path', default='./log/deeplabv3p.pth')
parser.add_argument('--config_path',  type=str,
                    help='config path', default='baseline_loveda.deeplabv3p')
parser.add_argument('--convert_path',  type=str,
                    help='convert_path', default='./convert.pth')

args = parser.parse_args()

seed_torch(2333)
logger = logging.getLogger(__name__)

er.registry.register_all()


def evaluate(ckpt_path, config_path='base.hrnetw32'):
    cfg = import_config(config_path)
    model_state_dict = load_model_state_dict_from_ckpt(ckpt_path)

    log_dir = os.path.dirname(ckpt_path)
    test_dataloader = make_dataloader(cfg['data']['test'])
    model = make_model(cfg['model'])
    deled_state = dict(model=dict())
    for k,v in model_state_dict.items():
        if k in model.state_dict().keys():
            deled_state['model'][k] = v
    #model.load_state_dict(model_state_dict, strict=True)
    #model.cuda()
    #model.eval()
    
    torch.save(deled_state, args.convert_path)
    

if __name__ == '__main__':
    # ckpt_path = './log/deeplabv3p.pth'
    # config_path = 'baseline_loveda.deeplabv3p'
    evaluate(args.ckpt_path, args.config_path)
