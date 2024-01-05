import glob
import os
from skimage.io import imsave,imread
import numpy as np
from tqdm import tqdm
import math
from torch.nn.modules.utils import _pair
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default=r'./FloodNet-Supervised_v1.0/train')
parser.add_argument('--save_dir', type=str, default=r'./FloodNet-Supervised_v1.0/train')

args = parser.parse_args()


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


def clip_floodnet(img_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    imgp_list = glob.glob(os.path.join(img_dir, '*.jpg'))
    clip_size = 1024
    stride = 1024
    save_dir = '/data1/wjj/FloodNet/FloodNet-Supervised_v1.0/test/image_p1024s1024'
    os.makedirs(save_dir, exist_ok=True)

    for imgp in tqdm(imgp_list):
        raw_image = imread(imgp)
        fname = os.path.basename(imgp)
        h, w, _ = raw_image.shape
        boxes = sliding_window((h,w), clip_size, stride)
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            clipped_image = raw_image[y_min:y_max, x_min:x_max].astype(np.uint8)
            save_path = os.path.join(save_dir, '%s_%d_%d_%d_%d.jpg' % (fname.replace('.jpg', ''), x_min, y_min, x_max, y_max))
            imsave(save_path, clipped_image, check_contrast=False)


if __name__ == '__main__':
    img_dir = os.path.join(args.root_dir, 'train-org-img')
    img_save_dir = os.path.join(args.root_dir, 'image_p1024s1024')
    clip_floodnet(img_dir, img_save_dir)

    img_dir = os.path.join(args.root_dir, 'train-label-img')
    img_save_dir = os.path.join(args.root_dir, 'label_p1024s1024')
    clip_floodnet(img_dir, img_save_dir)