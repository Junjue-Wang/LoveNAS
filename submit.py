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
from skimage.io import imsave
import argparse

parser = argparse.ArgumentParser(description='Infer methods')
parser.add_argument('--ckpt_path',  type=str,
                    help='ckpt path', default='./log/deeplabv3p.pth')
parser.add_argument('--config_path',  type=str,
                    help='config path', default='baseline_loveda.deeplabv3p')
parser.add_argument('--submit_dir',  type=str,
                    help='submit_dir', default='./submit_dir')
parser.add_argument('--tta',  type=bool,
                    help='use tta', default=False)
args = parser.parse_args()

logger = logging.getLogger(__name__)

er.registry.register_all()


def evaluate(ckpt_path, config_path='base.hrnetw32', use_tta=False):
    cfg = import_config(config_path)
    model_state_dict = load_model_state_dict_from_ckpt(ckpt_path)

    log_dir = os.path.dirname(ckpt_path)
    test_dataloader = make_dataloader(cfg['data']['test'])
    model = make_model(cfg['model'])
    
    model.load_state_dict(model_state_dict, strict=True)
    model.cuda()
    model.eval()

    vis_dir = os.path.join(log_dir, 'vis-{}'.format(os.path.basename(ckpt_path)))
    palette = np.array(list(COLOR_MAP.values())).reshape(-1).tolist()
    viz_op = er.viz.VisualizeSegmm(vis_dir, palette)
    os.makedirs(args.submit_dir, exist_ok=True)
    with torch.no_grad():
        for img, gt in tqdm(test_dataloader):
            img = img.cuda()
            if use_tta:
                pred = tta(model, img, tta_config=[
                    Scale(scale_factor=0.5),
                    Scale(scale_factor=0.75),
                    Scale(scale_factor=1.0),
                    Scale(scale_factor=1.25),
                    Scale(scale_factor=1.5),
                    Scale(scale_factor=1.75),
                    #HorizontalFlip(),
                    #VerticalFlip(),
                    #Rotate90k(1),
                    #Rotate90k(2),
                    #Rotate90k(3)
                ])
            else:
                pred = model(img)
            pred = pred.argmax(dim=1).cpu()

            for clsmap, imname in zip(pred, gt['fname']):
                res_idx = clsmap.cpu().numpy().astype(np.uint8)
                viz_op(res_idx, imname)
                imsave(os.path.join(args.submit_dir, imname), res_idx)
    torch.cuda.empty_cache()


if __name__ == '__main__':
    evaluate(args.ckpt_path, args.config_path, args.tta)
