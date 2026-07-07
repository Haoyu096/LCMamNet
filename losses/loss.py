import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F


class SoftIoULoss(nn.Module):
    def __init__(self, size_average=True):
        super(SoftIoULoss, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        smooth = 1
        intersection = pred * target
        loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() - intersection.sum() + smooth)
        loss = 1 - loss.mean()
        return loss


class BCEAndSoftIoULoss(nn.Module):
    def __init__(self, bce_weight=0.5, iou_weight=0.5, pos_weight=None, size_average=True):
        super(BCEAndSoftIoULoss, self).__init__()
        self.soft_iou = SoftIoULoss(size_average=size_average)
        if pos_weight is not None:
            pos_weight = torch.as_tensor([float(pos_weight)], dtype=torch.float32)
        self.bce_loss = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
            reduction='mean' if size_average else 'sum',
        )
        self.bce_weight = float(bce_weight)
        self.iou_weight = float(iou_weight)

    def forward(self, pred, target):
        loss_bce = self.bce_loss(pred, target)
        loss_iou = self.soft_iou(pred, target)
        return self.bce_weight * loss_bce + self.iou_weight * loss_iou


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        dims = tuple(range(1, pred.dim()))
        tp = torch.sum(pred * target, dim=dims)
        fp = torch.sum(pred * (1 - target), dim=dims)
        fn = torch.sum((1 - pred) * target, dim=dims)
        score = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - score.mean()


class OversizeLoss(nn.Module):
    """Penalize systematic over-segmentation more than under-segmentation."""
    def __init__(self, smooth=1.0):
        super(OversizeLoss, self).__init__()
        self.smooth = float(smooth)

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        dims = tuple(range(1, pred.dim()))
        pred_area = torch.sum(pred, dim=dims)
        target_area = torch.sum(target, dim=dims)
        oversize = torch.relu(pred_area - target_area)
        loss = oversize / (target_area + self.smooth)
        return loss.mean()


class BCEAndTverskyOversizeLoss(nn.Module):
    def __init__(
        self,
        bce_weight=0.4,
        tversky_weight=0.5,
        oversize_weight=0.1,
        pos_weight=None,
        alpha=0.7,
        beta=0.3,
        size_average=True,
    ):
        super(BCEAndTverskyOversizeLoss, self).__init__()
        if pos_weight is not None:
            pos_weight = torch.as_tensor([float(pos_weight)], dtype=torch.float32)
        self.bce_loss = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
            reduction='mean' if size_average else 'sum',
        )
        self.tversky_loss = TverskyLoss(alpha=alpha, beta=beta)
        self.oversize_loss = OversizeLoss()
        self.bce_weight = float(bce_weight)
        self.tversky_weight = float(tversky_weight)
        self.oversize_weight = float(oversize_weight)

    def forward(self, pred, target):
        loss_bce = self.bce_loss(pred, target)
        loss_tversky = self.tversky_loss(pred, target)
        loss_oversize = self.oversize_loss(pred, target)
        return (
            self.bce_weight * loss_bce
            + self.tversky_weight * loss_tversky
            + self.oversize_weight * loss_oversize
        )


def lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if gt_sorted.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def flatten_binary_scores(scores, labels, ignore=None):
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    return scores[valid], labels[valid]


class LovaszHingeLoss(nn.Module):
    def __init__(self, per_image=True):
        super(LovaszHingeLoss, self).__init__()
        self.per_image = bool(per_image)

    def _lovasz_hinge_flat(self, logits, labels):
        if labels.numel() == 0:
            return logits.sum() * 0.0
        signs = 2.0 * labels.float() - 1.0
        errors = 1.0 - logits * signs
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.detach()
        gt_sorted = labels[perm]
        grad = lovasz_grad(gt_sorted)
        return torch.dot(torch.relu(errors_sorted), grad)

    def forward(self, pred, target):
        if self.per_image:
            losses = []
            for logit, label in zip(pred, target):
                logit_flat, label_flat = flatten_binary_scores(logit, label)
                losses.append(self._lovasz_hinge_flat(logit_flat, label_flat))
            return torch.stack(losses).mean()
        pred_flat, target_flat = flatten_binary_scores(pred, target)
        return self._lovasz_hinge_flat(pred_flat, target_flat)


class BCEAndLovaszLoss(nn.Module):
    def __init__(self, bce_weight=0.7, lovasz_weight=0.3, pos_weight=None, size_average=True):
        super(BCEAndLovaszLoss, self).__init__()
        if pos_weight is not None:
            pos_weight = torch.as_tensor([float(pos_weight)], dtype=torch.float32)
        self.bce_loss = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
            reduction='mean' if size_average else 'sum',
        )
        self.lovasz_loss = LovaszHingeLoss(per_image=True)
        self.bce_weight = float(bce_weight)
        self.lovasz_weight = float(lovasz_weight)

    def forward(self, pred, target):
        loss_bce = self.bce_loss(pred, target)
        loss_lovasz = self.lovasz_loss(pred, target)
        return self.bce_weight * loss_bce + self.lovasz_weight * loss_lovasz


def _soft_iou_from_probs(prob, target, smooth=1.0):
    intersection = prob * target
    union = prob.sum() + target.sum() - intersection.sum()
    score = (intersection.sum() + smooth) / (union + smooth)
    return 1.0 - score


class MultiScaleSoftIoULoss(nn.Module):
    def __init__(self, pool_scales=(2, 4, 8), smooth=1.0):
        super(MultiScaleSoftIoULoss, self).__init__()
        self.pool_scales = tuple(int(s) for s in pool_scales)
        self.smooth = float(smooth)

    def forward(self, pred, target):
        prob = torch.sigmoid(pred)
        losses = []
        for scale in self.pool_scales:
            pooled_prob = F.avg_pool2d(prob, kernel_size=scale, stride=scale, ceil_mode=False)
            pooled_target = F.avg_pool2d(target, kernel_size=scale, stride=scale, ceil_mode=False)
            losses.append(_soft_iou_from_probs(pooled_prob, pooled_target, smooth=self.smooth))
        return torch.stack(losses).mean()


class BoundaryConsistencyLoss(nn.Module):
    def __init__(self):
        super(BoundaryConsistencyLoss, self).__init__()
        lap = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
        self.register_buffer('lap_kernel', lap.view(1, 1, 3, 3))

    def _edge_map(self, x):
        edge = F.conv2d(x, self.lap_kernel, padding=1)
        edge = torch.abs(edge)
        return torch.tanh(2.0 * edge)

    def forward(self, pred, target):
        prob = torch.sigmoid(pred)
        pred_edge = self._edge_map(prob)
        target_edge = self._edge_map(target)
        return F.smooth_l1_loss(pred_edge, target_edge, reduction='mean')


class StructuredSoftIoULoss(nn.Module):
    def __init__(
        self,
        global_weight=0.5,
        multiscale_weight=0.3,
        boundary_weight=0.2,
        pool_scales=(2, 4, 8),
        smooth=1.0,
    ):
        super(StructuredSoftIoULoss, self).__init__()
        self.global_weight = float(global_weight)
        self.multiscale_weight = float(multiscale_weight)
        self.boundary_weight = float(boundary_weight)
        self.global_iou = SoftIoULoss(size_average=True)
        self.multiscale_iou = MultiScaleSoftIoULoss(pool_scales=pool_scales, smooth=smooth)
        self.boundary_loss = BoundaryConsistencyLoss()

    def forward(self, pred, target):
        loss_global = self.global_iou(pred, target)
        loss_multiscale = self.multiscale_iou(pred, target)
        loss_boundary = self.boundary_loss(pred, target)
        return (
            self.global_weight * loss_global
            + self.multiscale_weight * loss_multiscale
            + self.boundary_weight * loss_boundary
        )


class EpochSwitchLoss(nn.Module):
    def __init__(self, loss_before, loss_after, switch_epoch):
        super(EpochSwitchLoss, self).__init__()
        self.loss_before = loss_before
        self.loss_after = loss_after
        self.switch_epoch = int(switch_epoch)
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def forward(self, pred, target):
        if self.current_epoch >= self.switch_epoch:
            return self.loss_after(pred, target)
        return self.loss_before(pred, target)


class BCELoss(nn.Module):
    """
    Binary Cross Entropy Loss wrapper using BCEWithLogitsLoss for numerical stability with AMP.
    Assumes input 'pred' is the raw Logits (unactivated scores).
    """
    def __init__(self, size_average=True, pos_weight=None):
        super(BCELoss, self).__init__()
        if pos_weight is not None:
            pos_weight = torch.as_tensor([float(pos_weight)], dtype=torch.float32)
        self.bce_loss = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
            reduction='mean' if size_average else 'sum'
        )

    def forward(self, pred, target):
        return self.bce_loss(pred, target)


class SLSIoULoss(nn.Module):
    def __init__(self, warm_epoch=5, with_shape=True):
        super(SLSIoULoss, self).__init__()
        self.warm_epoch = int(warm_epoch)
        self.with_shape = bool(with_shape)
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def forward(self, pred_log, target):
        pred = torch.sigmoid(pred_log)
        smooth = 0.0

        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        target_sum = torch.sum(target, dim=(1, 2, 3))

        dis = torch.pow((pred_sum - target_sum) / 2, 2)
        alpha = (torch.min(pred_sum, target_sum) + dis + smooth) / (torch.max(pred_sum, target_sum) + dis + smooth)

        loss = (intersection_sum + smooth) / (pred_sum + target_sum - intersection_sum + smooth)
        beta = torch.sum(target, dim=(1, 2, 3)) / (torch.sum(target) + np.spacing(1))
        lloss = LLoss(pred, target)

        if self.current_epoch > self.warm_epoch:
            siou_loss = alpha * loss
            if self.with_shape:
                loss = (beta * (1 - siou_loss)).sum() + lloss
            else:
                loss = (beta * (1 - siou_loss)).sum()
        else:
            loss = (beta * (1 - loss)).sum()
        return loss


def LLoss(pred, target):
    loss = torch.tensor(0.0, device=pred.device)

    patch_size = pred.shape[0]
    h = pred.shape[2]
    w = pred.shape[3]
    x_index = torch.arange(0, w, 1, device=pred.device, dtype=pred.dtype).view(1, 1, w).repeat((1, h, 1)) / w
    y_index = torch.arange(0, h, 1, device=pred.device, dtype=pred.dtype).view(1, h, 1).repeat((1, 1, w)) / h
    smooth = 1e-8
    for i in range(patch_size):
        pred_centerx = (x_index * pred[i]).mean()
        pred_centery = (y_index * pred[i]).mean()

        target_centerx = (x_index * target[i]).mean()
        target_centery = (y_index * target[i]).mean()

        angle_loss = (4 / (torch.pi ** 2)) * (
            torch.square(
                torch.arctan((pred_centery) / (pred_centerx + smooth))
                - torch.arctan((target_centery) / (target_centerx + smooth))
            )
        )
        pred_length = torch.sqrt(pred_centerx * pred_centerx + pred_centery * pred_centery + smooth)
        target_length = torch.sqrt(target_centerx * target_centerx + target_centery * target_centery + smooth)

        length_loss = (torch.min(pred_length, target_length)) / (torch.max(pred_length, target_length) + smooth)
        loss = loss + (1 - length_loss + angle_loss) / patch_size

    return loss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _build_base_loss(mode, pos_weight=None, **kwargs):
    mode = str(mode).lower()
    if mode in ('softiou', 'iou'):
        return SoftIoULoss()
    if mode in ('structured_softiou', 'struct_softiou', 'ssiou'):
        return StructuredSoftIoULoss(
            global_weight=float(kwargs.get('structured_softiou_global_weight', 0.5)),
            multiscale_weight=float(kwargs.get('structured_softiou_multiscale_weight', 0.3)),
            boundary_weight=float(kwargs.get('structured_softiou_boundary_weight', 0.2)),
            pool_scales=tuple(kwargs.get('structured_softiou_pool_scales', (2, 4, 8))),
        )
    if mode == 'bce':
        return BCELoss(size_average=True, pos_weight=pos_weight)
    if mode in ('bce_softiou', 'softiou_bce'):
        return BCEAndSoftIoULoss(bce_weight=0.5, iou_weight=0.5, pos_weight=pos_weight, size_average=True)
    if mode in ('bce70_softiou30', 'bce_softiou_7030'):
        return BCEAndSoftIoULoss(bce_weight=0.7, iou_weight=0.3, pos_weight=pos_weight, size_average=True)
    if mode in ('bce30_softiou70', 'bce_softiou_3070'):
        return BCEAndSoftIoULoss(bce_weight=0.3, iou_weight=0.7, pos_weight=pos_weight, size_average=True)
    if mode in ('tversky', 'tversky70_30'):
        return TverskyLoss(alpha=0.7, beta=0.3)
    if mode in ('bce_tversky_oversize', 'bto', 'bce40_tversky50_area10'):
        return BCEAndTverskyOversizeLoss(
            bce_weight=0.4,
            tversky_weight=0.5,
            oversize_weight=0.1,
            pos_weight=pos_weight,
            alpha=0.7,
            beta=0.3,
            size_average=True,
        )
    if mode in ('bce_lovasz', 'bce70_lovasz30', 'lovasz_bce'):
        return BCEAndLovaszLoss(
            bce_weight=float(kwargs.get('bce_lovasz_bce_weight', 0.7)),
            lovasz_weight=float(kwargs.get('bce_lovasz_lovasz_weight', 0.3)),
            pos_weight=pos_weight,
            size_average=True,
        )
    if mode == 'sls':
        return SLSIoULoss(warm_epoch=5, with_shape=True)
    raise ValueError(f'Unknown loss_mode: {mode}')


def build_loss(args):
    mode = str(getattr(args, 'loss_mode', 'softiou')).lower()
    pos_weight = getattr(args, 'bce_pos_weight', None)
    extra_kwargs = {}
    if hasattr(args, 'bce_lovasz_bce_weight'):
        extra_kwargs['bce_lovasz_bce_weight'] = getattr(args, 'bce_lovasz_bce_weight')
    if hasattr(args, 'bce_lovasz_lovasz_weight'):
        extra_kwargs['bce_lovasz_lovasz_weight'] = getattr(args, 'bce_lovasz_lovasz_weight')
    if hasattr(args, 'structured_softiou_global_weight'):
        extra_kwargs['structured_softiou_global_weight'] = getattr(args, 'structured_softiou_global_weight')
    if hasattr(args, 'structured_softiou_multiscale_weight'):
        extra_kwargs['structured_softiou_multiscale_weight'] = getattr(args, 'structured_softiou_multiscale_weight')
    if hasattr(args, 'structured_softiou_boundary_weight'):
        extra_kwargs['structured_softiou_boundary_weight'] = getattr(args, 'structured_softiou_boundary_weight')
    if hasattr(args, 'structured_softiou_pool_scales'):
        extra_kwargs['structured_softiou_pool_scales'] = getattr(args, 'structured_softiou_pool_scales')

    if mode in ('schedule_bce30_softiou70_to_bce_softiou', 'sched_bce30_to_bce50'):
        switch_epoch = int(getattr(args, 'loss_switch_epoch', 400))
        loss = EpochSwitchLoss(
            _build_base_loss('bce30_softiou70', pos_weight=pos_weight, **extra_kwargs),
            _build_base_loss('bce_softiou', pos_weight=pos_weight, **extra_kwargs),
            switch_epoch=switch_epoch,
        )
        return loss.cuda()

    loss = _build_base_loss(mode, pos_weight=pos_weight, **extra_kwargs)
    if hasattr(loss, 'warm_epoch') and hasattr(args, 'warmup_epochs'):
        loss.warm_epoch = int(getattr(args, 'warmup_epochs', loss.warm_epoch))
    return loss.cuda()
