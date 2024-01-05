import random
import numpy as np
from albumentations import DualTransform
import math
__all__ = [
    'DivisiblePad',
]


class DivisiblePad(DualTransform):
    def __init__(self,
                 size_divisor=32,
                 value=0,
                 mask_value=-1,
                 always_apply=False,
                 p=1.0):
        super(DivisiblePad, self).__init__(always_apply, p)
        self.size_divisor = size_divisor
        self.value = value
        self.mask_value = mask_value


    def apply(self, img, **params):
        height, width, _ = img.shape

        nheight = math.ceil(height / self.size_divisor) * self.size_divisor
        nwidth = math.ceil(width / self.size_divisor) * self.size_divisor
        pad_bottom = nwidth - width
        pad_right = nheight - height
        return np.pad(img, ((0, pad_right), (0, pad_bottom), (0, 0)), mode='constant',
                      constant_values=self.value)

    def apply_to_mask(self, img, **params):

        height, width = img.shape

        nheight = math.ceil(height / self.size_divisor) * self.size_divisor
        nwidth = math.ceil(width / self.size_divisor) * self.size_divisor
        pad_bottom = nwidth - width
        pad_right = nheight - height

        return np.pad(img, ((0, pad_right), (0, pad_bottom)), mode='constant',
                      constant_values=self.mask_value)

    def get_transform_init_args_names(self):
        return ("size_divisor", "value", "mask_value")

if __name__ == '__main__':
    x = np.ones((245, 245, 3))
    mask = np.ones((245, 245))
    o = DivisiblePad()(image=x, mask=mask)
    print(o['image'].shape)
    print(o['mask'].shape)