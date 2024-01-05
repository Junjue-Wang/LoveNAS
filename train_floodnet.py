import ever as er
import torch
import numpy as np
import os
from data.floodnet import COLOR_MAP
from tqdm import tqdm
import random
from eval_window import SegmSlidingWinInference
er.registry.register_all()


def evaluate_cls_fn(self, test_dataloader, config=None):
    self.model.eval()
    classes = self.model.module.config.classes if self.model.module.config.classes != 1 else 2
    metric_op = er.metric.PixelMetric(classes, logdir=self._model_dir, logger=self.logger)

    vis_dir = os.path.join(self._model_dir, 'vis-{}'.format(self.checkpoint.global_step))

    palette = np.array(list(COLOR_MAP.values())).reshape(-1).tolist()
    viz_op = er.viz.VisualizeSegmm(vis_dir, palette)
    segm_helper = SegmSlidingWinInference()
    with torch.no_grad():
        for img, gt in tqdm(test_dataloader):
            h, w = img.shape[2:4]
            seg_helper = segm_helper.patch((h, w), patch_size=(1024, 1024), stride=512,
                                           transforms=None)
            pred = seg_helper.forward(self.model, img, size_divisor=32)
            y_true = gt['cls']
            pred = pred.argmax(dim=1).cpu()
            valid_inds = y_true != -1
            metric_op.forward(y_true[valid_inds], pred[valid_inds])
            for clsmap, imname in zip(pred, gt['fname']):
                viz_op(clsmap.cpu().numpy().astype(np.uint8), imname.replace('jpg', 'png'))
    metric_op.summary_all()
    torch.cuda.empty_cache()


def register_evaluate_fn(launcher):
    launcher.override_evaluate(evaluate_cls_fn)



def seed_torch(seed=2333):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True



if __name__ == '__main__':
    seed_torch(2333)
    trainer = er.trainer.get_trainer('th_amp_ddp')()
    trainer.run(after_construct_launcher_callbacks=[register_evaluate_fn])
