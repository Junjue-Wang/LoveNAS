import ever as er
from ever.core.builder import make_model, make_dataloader
import torch
import numpy as np
import os
from data.loveda import COLOR_MAP
import logging
from ever.core.checkpoint import load_model_state_dict_from_ckpt
from ever.core.config import import_config
from train_loveda import seed_torch
import argparse
from albumentations import Compose, Normalize
# from ever.magic.bigimage import sliding_window
from tqdm import tqdm
import math
from torch.nn.modules.utils import _pair




seed_torch(2333)
logger = logging.getLogger(__name__)

er.registry.register_all()


def sliding_window(input_size, kernel_size, stride):
    ih, iw = input_size
    kh, kw = _pair(kernel_size)
    sh, sw = _pair(stride)
    assert ih > 0 and iw > 0 and kh > 0 and kw > 0 and sh > 0 and sw > 0

    kh = ih if kh > ih else kh
    kw = iw if kw > iw else kw

    num_rows = math.ceil((ih - kh) / sh) if math.ceil((ih - kh) / sh) * sh + kh >= ih else math.ceil(
        (ih - kh) / sh) + 1
    num_cols = math.ceil((iw - kw) / sw) if math.ceil((iw - kw) / sw) * sw + kw >= iw else math.ceil(
        (iw - kw) / sw) + 1

    x, y = np.meshgrid(np.arange(num_cols + 1), np.arange(num_rows + 1))
    xmin = x * sw
    ymin = y * sh

    xmin = xmin.ravel()
    ymin = ymin.ravel()
    xmin_offset = np.where(xmin + kw > iw, iw - xmin - kw, np.zeros_like(xmin))
    ymin_offset = np.where(ymin + kh > ih, ih - ymin - kh, np.zeros_like(ymin))
    boxes = np.stack([xmin + xmin_offset, ymin + ymin_offset,
                      np.minimum(xmin + kw, iw), np.minimum(ymin + kh, ih)], axis=1)

    return boxes


class SegmSlidingWinInference(object):
    def __init__(self):
        super(SegmSlidingWinInference, self).__init__()
        self._h = None
        self._w = None
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def patch(self, input_size, patch_size, stride, transforms=None):
        """ divide large image into small patches.

        Returns:

        """
        self.wins = sliding_window(input_size, patch_size, stride)
        self.transforms = transforms
        return self

    def merge(self, out_list):
        pred_list, win_list = list(zip(*out_list))
        num_classes = pred_list[0].size(1)
        res_img = torch.zeros(pred_list[0].size(0), num_classes, self._h, self._w, dtype=torch.float32)
        res_count = torch.zeros(self._h, self._w, dtype=torch.float32)

        for pred, win in zip(pred_list, win_list):
            res_count[win[1]:win[3], win[0]: win[2]] += 1
            res_img[:, :, win[1]:win[3], win[0]: win[2]] += pred.cpu()

        avg_res_img = res_img / res_count

        return avg_res_img

    def forward(self, model, image_tensor, **kwargs):
        assert self.wins is not None, 'patch must be performed before forward.'
        # set the image height and width
        self._h, self._w = image_tensor.shape[2:4]
        return self._forward(model, image_tensor, **kwargs)

    def _forward(self, model, image_tensor, **kwargs):
        self.device = kwargs.get('device', self.device)
        size_divisor = kwargs.get('size_divisor', None)
        assert self.wins is not None, 'patch must be performed before forward.'
        out_list = []
        for win in tqdm(self.wins):
            x1, y1, x2, y2 = win
            image = image_tensor[: ,:, y1:y2, x1:x2]
            if self.transforms is not None:
                image = self.transforms(image=image)['image']
            h, w = image.shape[2:4]
            if size_divisor is not None:
                image = er.preprocess.function.th_divisible_pad(image, size_divisor)
            image = image.to(self.device)
            with torch.no_grad():
                out = model(image)
            if size_divisor is not None:
                out = out[:, :, :h, :w]
            out_list.append((out.cpu(), win))
            torch.cuda.empty_cache()
        self.wins = None

        return self.merge(out_list)

def evaluate(ckpt_path, config_path='base.hrnetw32', use_tta=False):
    cfg = import_config(config_path)
    model_state_dict = load_model_state_dict_from_ckpt(ckpt_path)

    log_dir = os.path.dirname(ckpt_path)
    test_dataloader = make_dataloader(cfg['data']['test'])
    model = make_model(cfg['model'])
    model.load_state_dict(model_state_dict)
    model.cuda()
    model.eval()

    metric_op = er.metric.PixelMetric(7, logdir=log_dir, logger=logger)
    vis_dir = os.path.join(log_dir, 'vis-{}'.format(os.path.basename(ckpt_path)))
    palette = np.array(list(COLOR_MAP.values())).reshape(-1).tolist()
    viz_op = er.viz.VisualizeSegmm(vis_dir, palette)
    segm_helper = SegmSlidingWinInference()

    with torch.no_grad():
        for idx, (img, gt) in enumerate(test_dataloader):
            h, w = img.shape[2:4]
            logging.info('Progress - [{} / {}] size = ({}, {})'.format(idx + 1, len(test_dataloader), h, w))
            seg_helper = segm_helper.patch((h, w), patch_size=(args.patch_size, args.patch_size), stride=args.stride,
                                       transforms=None)
            pred = seg_helper.forward(model, img, size_divisor=32)
            
            y_true = gt['cls']
            y_true = y_true.cpu()
            
            pred = pred.argmax(dim=1).cpu()

            valid_inds = y_true != -1
            metric_op.forward(y_true[valid_inds], pred[valid_inds])

            for clsmap, imname in zip(pred, gt['fname']):
                viz_op(clsmap.cpu().numpy().astype(np.uint8), imname.replace('tif', 'png'))
    metric_op.summary_all()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    # ckpt_path = './log/deeplabv3p.pth'
    # config_path = 'baseline_loveda.deeplabv3p'
    parser = argparse.ArgumentParser(description='Eval methods')
    parser.add_argument('--ckpt_path',  type=str,
                        help='ckpt path', default='./log/deeplabv3p.pth')
    parser.add_argument('--config_path',  type=str,
                        help='config path', default='baseline_loveda.deeplabv3p')
    parser.add_argument('--tta',  type=bool,
                        help='use tta', default=False)
    parser.add_argument('--patch_size',  type=int,
                        help='patch_size', default=512)
    parser.add_argument('--stride',  type=int,
                        help='stride', default=256)
    args = parser.parse_args()
    evaluate(args.ckpt_path, args.config_path, args.tta)
