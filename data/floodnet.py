import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import Dataset
import glob
import os
from skimage.io import imread
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, Normalize, RandomCrop, RandomScale
from albumentations import OneOf, Compose
import ever as er
from collections import OrderedDict
from ever.interface import ConfigurableMixin
from torch.utils.data import SequentialSampler
from ever.api.data import distributed, CrossValSamplerGenerator
import numpy as np
import logging

logger = logging.getLogger(__name__)

COLOR_MAP = OrderedDict(
    background=(255, 255, 255),
    building_flooded=(255, 0, 0),
    building_no_flooded=(61,230,250),
    road_flooded=(243, 118, 74),
    road_non_flooded=(255, 255, 0),
    water=(0, 0, 255),
    tree=(0, 255, 0),
    vehicle=(255,0,245),
    pool=(95,198,201),
    grass=(51,153,51),
)



class FloodNetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None):
        self.rgb_filepath_list = []
        self.cls_filepath_list= []
        if isinstance(image_dir, list):
            for img_dir_path, mask_dir_path in zip(image_dir, mask_dir):
                self.batch_generate(img_dir_path, mask_dir_path)

        else:
            self.batch_generate(image_dir, mask_dir)

        self.transforms = transforms

    def batch_generate(self, image_dir, mask_dir):
        rgb_filepath_list = glob.glob(os.path.join(image_dir, '*.jpg'))

        logger.info('Dataset images: %d' % len(rgb_filepath_list))
        rgb_filename_list = [os.path.split(fp)[-1].split('.')[0] for fp in rgb_filepath_list]
        cls_filepath_list = []
        if mask_dir is not None:
            for fname in rgb_filename_list:
                if '_' in fname:
                    cls_filepath_list.append(os.path.join(mask_dir, f'{fname}.png'))
                else:
                    cls_filepath_list.append(os.path.join(mask_dir, f'{fname}_lab.png'))
        self.rgb_filepath_list += rgb_filepath_list
        self.cls_filepath_list += cls_filepath_list

    def __getitem__(self, idx):
        image = imread(self.rgb_filepath_list[idx])
        mask = imread(self.cls_filepath_list[idx]).astype(np.long)
        if self.transforms is not None:
            blob = self.transforms(image=image, mask=mask)
            image = blob['image']
            mask = blob['mask']

        return image, dict(cls=mask, fname=os.path.basename(self.rgb_filepath_list[idx]))

    def __len__(self):
        return len(self.rgb_filepath_list)


@er.registry.DATALOADER.register()
class FloodNetLoader(DataLoader, ConfigurableMixin):
    def __init__(self, config):
        ConfigurableMixin.__init__(self, config)
        dataset = FloodNetDataset(self.config.image_dir, self.config.mask_dir, self.config.transforms)
        if self.config.CV.i != -1:
            CV = CrossValSamplerGenerator(dataset, distributed=True, seed=2333)
            sampler_pairs = CV.k_fold(self.config.CV.k)
            train_sampler, val_sampler = sampler_pairs[self.config.CV.i]
            if self.config.training:
                sampler = train_sampler
            else:
                sampler = val_sampler
        else:
            sampler = distributed.StepDistributedSampler(dataset) if self.config.training else SequentialSampler(
                dataset)

        super(FloodNetLoader, self).__init__(dataset,
                                       self.config.batch_size,
                                       sampler=sampler,
                                       num_workers=self.config.num_workers,
                                       pin_memory=True)
    def set_default_config(self):
        self.config.update(dict(
            image_dir=None,
            mask_dir=None,
            batch_size=4,
            num_workers=4,
            transforms=Compose([
                OneOf([
                    HorizontalFlip(True),
                    VerticalFlip(True),
                    RandomRotate90(True),
                ], p=0.75),
                Normalize(mean=(), std=(), max_pixel_value=1, always_apply=True),
                ToTensorV2()
            ]),
        ))


