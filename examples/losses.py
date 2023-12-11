import torch.nn as nn
from wilds.common.metrics.loss import ElementwiseLoss, Loss, MultiTaskLoss
from wilds.common.metrics.all_metrics import MSE
from utils import cross_entropy_with_logits_loss, LogitNormLoss, FocalLoss,mse_logitNormLoss

def initialize_loss(loss, config):
    if loss == 'cross_entropy':
        return ElementwiseLoss(loss_fn=nn.CrossEntropyLoss(reduction='none', ignore_index=-100))

    elif loss == 'lm_cross_entropy':
        return MultiTaskLoss(loss_fn=nn.CrossEntropyLoss(reduction='none', ignore_index=-100))

    elif loss == 'mse':
        return MSE(name='loss')

    elif loss == 'mse_logitNormLoss':
        return ElementwiseLoss(name='loss', loss_fn=mse_logitNormLoss)

    elif loss == 'multitask_bce':
        return MultiTaskLoss(loss_fn=nn.BCEWithLogitsLoss(reduction='none'))

    elif loss == 'fasterrcnn_criterion':
        from models.detection.fasterrcnn import FasterRCNNLoss
        return ElementwiseLoss(loss_fn=FasterRCNNLoss(config.device))

    elif loss == 'cross_entropy_logits':
        return ElementwiseLoss(loss_fn=cross_entropy_with_logits_loss)

    if loss == 'FocalLoss':
        return ElementwiseLoss(loss_fn=FocalLoss())

    if loss == 'logitNorm_loss':
        return ElementwiseLoss(loss_fn=LogitNormLoss(t=config.loss_kwargs['t']))
    else:
        raise ValueError(f'loss {loss} not recognized')
