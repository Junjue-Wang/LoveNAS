import torch.nn.functional as F
import torch.nn as nn
import torch

def _masked_ignore(y_pred: torch.Tensor, y_true: torch.Tensor, ignore_index: int):
    # usually used for BCE-like loss
    y_pred = y_pred.reshape((-1,))
    y_true = y_true.reshape((-1,))
    valid = y_true != ignore_index
    y_true = y_true.masked_select(valid).float()
    y_pred = y_pred.masked_select(valid).float()
    return y_pred, y_true

def tversky_loss_with_logits(y_pred: torch.Tensor, y_true: torch.Tensor, alpha: float, beta: float,
                             smooth_value: float = 1.0,
                             ignore_index: int = 255):
    y_pred, y_true = _masked_ignore(y_pred, y_true, ignore_index)

    y_pred = y_pred.sigmoid()
    tp = (y_pred * y_true).sum()
    # fp = (y_pred * (1 - y_true)).sum()
    fp = y_pred.sum() - tp
    # fn = ((1 - y_pred) * y_true).sum()
    fn = y_true.sum() - tp

    tversky_coeff = (tp + smooth_value) / (tp + alpha * fn + beta * fp + smooth_value)
    return 1. - tversky_coeff

def binary_cross_entropy_with_logits(output: torch.Tensor, target: torch.Tensor, reduction: str = 'mean',
                                     ignore_index: int = 255):
    output, target = _masked_ignore(output, target, ignore_index)
    return F.binary_cross_entropy_with_logits(output, target, reduction=reduction)



def softmax_focalloss(y_pred, y_true, ignore_index=-1, gamma=2.0, normalize=False):
    """
    Args:
        y_pred: [N, #class, H, W]
        y_true: [N, H, W] from 0 to #class
        gamma: scalar
    Returns:
    """
    losses = F.cross_entropy(y_pred, y_true, ignore_index=ignore_index, reduction='none')
    with torch.no_grad():
        p = y_pred.softmax(dim=1)
        modulating_factor = (1 - p).pow(gamma)
        valid_mask = ~ y_true.eq(ignore_index)
        masked_y_true = torch.where(valid_mask, y_true, torch.zeros_like(y_true))
        modulating_factor = torch.gather(modulating_factor, dim=1, index=masked_y_true.unsqueeze(dim=1)).squeeze_(dim=1)
        scale = 1.
        if normalize:
            scale = losses.sum() / (losses * modulating_factor).sum()
    losses = scale * (losses * modulating_factor).sum() / (valid_mask.sum() + p.size(0))

    return losses

import ever.module as erm

def multi_binary_label(batched_mask: torch.Tensor, num_classes, ignore_index):
    labels = []
    for cls in range(0, num_classes):
        binary_label = torch.zeros_like(batched_mask)
        binary_label[batched_mask == cls] = 1
        binary_label[batched_mask == ignore_index] = ignore_index
        labels.append(binary_label.to(torch.long))
    return labels


def multi_binary_loss(y_pred, y_true, reduction='mean', ignore_index=-1):
    num_classes = y_pred.size(1)
    labels = multi_binary_label(y_true, num_classes, ignore_index=ignore_index)
    losses = []
    for cls in range(0, num_classes):
        bipred = y_pred[:, cls, :, :]
        bipred = bipred.reshape(y_pred.size(0), 1, y_pred.size(2), y_pred.size(3)).contiguous()
        bce_loss = erm.loss.label_smoothing_binary_cross_entropy(bipred,
                                                             labels[cls].reshape_as(bipred).float(), ignore_index=-1)
        dice_loss = erm.loss.dice_loss_with_logits(bipred, labels[cls], ignore_index=-1)
        losses.append(bce_loss+dice_loss)

    if 'sum' == reduction:
        tloss = sum(losses)
    elif 'mean' == reduction:
        tloss = sum(losses) / float(num_classes)
    else:
        raise ValueError()

    return tloss, losses

class SegmentationLoss(nn.Module):
    def __init__(self, loss_config):
        super(SegmentationLoss, self).__init__()
        self.loss_config = loss_config

    def forward(self, y_pred, y_true: torch.Tensor):
        loss_dict = dict()
        if 'ce' in self.loss_config:
            loss_dict['ce_loss'] = F.cross_entropy(y_pred, y_true.long(), ignore_index=-1)
        if 'fcloss' in self.loss_config:
            loss_dict['fc_loss'] = softmax_focalloss(y_pred, y_true, gamma=self.loss_config.fcloss.gamma, normalize=True)

        if 'bceloss' in self.loss_config:
            y_predb = y_pred[:, 0, :, :]
            invalidmask = y_true == -1
            bg_y_true = torch.where(y_true>0, torch.ones_like(y_predb), torch.zeros_like(y_predb))
            bg_y_true[invalidmask] = -1
            loss_dict['bceloss'] = binary_cross_entropy_with_logits(y_predb, bg_y_true, ignore_index=-1) * self.loss_config.bceloss.scaler


        if 'tverloss' in self.loss_config:
            y_predb = y_pred[:, 0, :, :]
            invalidmask = y_true == -1
            bg_y_true = torch.where(y_true>0, torch.ones_like(y_predb), torch.zeros_like(y_predb))
            bg_y_true[invalidmask] = -1
            loss_dict['tverloss'] = tversky_loss_with_logits(y_predb, bg_y_true, self.loss_config.tverloss.alpha,
                                                             self.loss_config.tverloss.beta, ignore_index=-1) * self.loss_config.tverloss.scaler

        if 'mbloss' in self.loss_config:
            loss_dict['mbloss'], losses = multi_binary_loss(y_pred, y_true, ignore_index=-1)
            for idx, l in enumerate(losses):
                loss_dict['loss_%d' % idx ] = l.data

        return loss_dict



if __name__ == '__main__':
    pred = torch.rand((2, 4, 10, 10))
    gt = torch.ones(2, 10, 10)
    gt[0, :, :] = -1
    loss = multi_binary_loss(pred, gt)