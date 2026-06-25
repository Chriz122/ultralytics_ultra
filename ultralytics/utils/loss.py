# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import CITYSCAPES_WEIGHT, OKS_SIGMA, RLE_WEIGHT
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist, rbox2dist


class AdaptiveThresholdFocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(AdaptiveThresholdFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        # self.reduction = loss_fcn.reduction
        # self.loss_fcn.reduction = 'none'  # required to apply FL to each element
 
    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)  # 得出预测概率
        p_t = torch.Tensor(p_t)  # 将张量转化为pytorch张量，使其在pytorch中可以进行张量运算
 
        mean_pt = p_t.mean()
        p_t_list = []
        p_t_list.append(mean_pt)
        p_t_old = sum(p_t_list) / len(p_t_list)
        p_t_new = 0.05 * p_t_old + 0.95 * mean_pt
        # gamma =2
        gamma = -torch.log(p_t_new)
        # 处理大于0.5的元素
        p_t_high = torch.where(p_t > 0.5, (1.000001 - p_t) ** gamma, torch.zeros_like(p_t))
 
        # 处理小于0.5的元素
        p_t_low = torch.where(p_t <= 0.5, (1.5 - p_t) ** (-torch.log(p_t)), torch.zeros_like(p_t))  # # 将两部分结果相加
        modulating_factor = p_t_high + p_t_low
        loss *= modulating_factor
        # if self.reduction == 'mean':
        #     return loss.mean()
        # elif self.reduction == 'sum':
        #     return loss.sum()
        # else:  # 'none'
        return loss


class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al.

    Implements the Varifocal Loss function for addressing class imbalance in object detection by focusing on
    hard-to-classify examples and balancing positive/negative samples.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (float): The balancing factor used to address class imbalance.

    References:
        https://arxiv.org/abs/2008.13367
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        """Initialize the VarifocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Compute varifocal loss between predictions and ground truth."""
        weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5).

    Implements the Focal Loss function for addressing class imbalance by down-weighting easy examples and focusing on
    hard negatives during training.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (torch.Tensor): The balancing factor used to address class imbalance.
    """

    def __init__(self, gamma: float = 1.5, alpha: float = 0.25):
        """Initialize FocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss with modulating factors for class imbalance."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if (self.alpha > 0).any():
            self.alpha = self.alpha.to(device=pred.device, dtype=pred.dtype)
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)."""

    def __init__(self, reg_max: int = 16) -> None:
        """Initialize the DFL module with regularization maximum."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return sum of left and right DFL losses from https://ieeexplore.ieee.org/document/9792391."""
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes."""

    def __init__(self, reg_max: int = 16, hyp=None): # ★ 新增 hyp
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.hyp = hyp # ★ 儲存 hyp

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
        hw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        
        # ★ 損失函數修改處：從 yaml 讀取 IoU 相關參數
        iou_type = getattr(self.hyp, 'iou_type', 'CIoU') if self.hyp else 'CIoU'
        GIoU = iou_type == 'GIoU'
        DIoU = iou_type == 'DIoU'
        CIoU = iou_type == 'CIoU'
        EIoU = iou_type == 'EIoU'
        SIoU = iou_type == 'SIoU'
        WIoU = iou_type == 'WIoU'
        ShapeIoU = iou_type == 'ShapeIoU'
        PIoU = iou_type == 'PIoU'
        PIoU2 = iou_type == 'PIoU2'
        mpdiou = iou_type == 'mpdiou'
        FoCIoU = iou_type == 'FoCIoU'
        
        Inner = getattr(self.hyp, 'Inner', False) if self.hyp else False
        Focaleriou = getattr(self.hyp, 'Focaleriou', False) if self.hyp else False
        d = getattr(self.hyp, 'd', 0.00) if self.hyp else 0.00
        u = getattr(self.hyp, 'u', 0.95) if self.hyp else 0.95
        ratio = getattr(self.hyp, 'ratio', 1.15) if self.hyp else 1.15
        Lambda = getattr(self.hyp, 'Lambda', 1.3) if self.hyp else 1.3
        scale = getattr(self.hyp, 'scale', 0.0) if self.hyp else 0.0

        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask],  xywh=False, 
                       GIoU=GIoU, DIoU=DIoU, CIoU=CIoU, EIoU=EIoU,
                       SIoU=SIoU, WIoU=WIoU, ShapeIoU=ShapeIoU, PIoU=PIoU, PIoU2=PIoU2, 
                       hw=hw[fg_mask], mpdiou=mpdiou, Inner=Inner,
                       Focaleriou=Focaleriou, FoCIoU=FoCIoU,
                       d=d, u=u, ratio=ratio, eps=1e-7, Lambda=Lambda, scale=scale)

        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            # normalize ltrb by image size
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


class RLELoss(nn.Module):
    """Residual Log-Likelihood Estimation Loss.

    Args:
        use_target_weight (bool): Option to use weighted loss.
        size_average (bool): Option to average the loss by the batch_size.
        residual (bool): Option to add L1 loss and let the flow learn the residual error distribution.

    References:
        https://arxiv.org/abs/2107.11291
        https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/losses/regression_loss.py
    """

    def __init__(self, use_target_weight: bool = True, size_average: bool = True, residual: bool = True):
        """Initialize RLELoss with target weight and residual options.

        Args:
            use_target_weight (bool): Whether to use target weights for loss calculation.
            size_average (bool): Whether to average the loss over elements.
            residual (bool): Whether to include residual log-likelihood term.
        """
        super().__init__()
        self.size_average = size_average
        self.use_target_weight = use_target_weight
        self.residual = residual

    def forward(
        self, sigma: torch.Tensor, log_phi: torch.Tensor, error: torch.Tensor, target_weight: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            sigma (torch.Tensor): Output sigma, shape (N, D).
            log_phi (torch.Tensor): Output log_phi, shape (N).
            error (torch.Tensor): Error, shape (N, D).
            target_weight (torch.Tensor): Weights across different joint types, shape (N).
        """
        log_sigma = torch.log(sigma)
        loss = log_sigma - log_phi.unsqueeze(1)

        if self.residual:
            loss += torch.log(sigma * 2) + torch.abs(error)

        if self.use_target_weight:
            assert target_weight is not None, "'target_weight' should not be None when 'use_target_weight' is True."
            if target_weight.dim() == 1:
                target_weight = target_weight.unsqueeze(1)
            loss *= target_weight

        if self.size_average:
            loss /= len(loss)

        return loss.sum()


class MLECCLoss(nn.Module):
    """Maximum Likelihood Estimation loss for Coordinate Classification."""
    def __init__(self, reduction: str = 'mean', mode: str = 'log', loss_weight: float = 1.0):
        super().__init__()
        self.reduction = reduction
        self.mode = mode
        self.loss_weight = loss_weight

    def forward(self, outputs, targets, target_weight=None):
        prob = 1.0
        for o, t in zip(outputs, targets):
            prob *= (o * t).sum(dim=-1)

        if self.mode == 'linear':
            loss = 1.0 - prob
        elif self.mode == 'square':
            loss = 1.0 - prob.pow(2)
        elif self.mode == 'log':
            loss = -torch.log(prob + 1e-4)

        loss[torch.isnan(loss)] = 0.0

        if target_weight is not None:
            for i in range(loss.ndim - target_weight.ndim):
                target_weight = target_weight.unsqueeze(-1)
            loss = loss * target_weight

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss * self.loss_weight


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses for rotated bounding boxes."""

    def __init__(self, reg_max: int):
        """Initialize the RotatedBboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for rotated bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = rbox2dist(
                target_bboxes[..., :4], anchor_points, target_bboxes[..., 4:5], reg_max=self.dfl_loss.reg_max - 1
            )
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = rbox2dist(target_bboxes[..., :4], anchor_points, target_bboxes[..., 4:5])
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


class MultiChannelDiceLoss(nn.Module):
    """Criterion class for computing multi-channel Dice losses."""

    def __init__(self, smooth: float = 1e-6, reduction: str = "mean"):
        """Initialize MultiChannelDiceLoss with smoothing and reduction options.

        Args:
            smooth (float): Smoothing factor to avoid division by zero.
            reduction (str): Reduction method ('mean', 'sum', or 'none').
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate multi-channel Dice loss between predictions and targets."""
        assert pred.size() == target.size(), "the size of predict and target must be equal."

        pred = pred.sigmoid()
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice
        dice_loss = dice_loss.mean(dim=1)

        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


class BCEDiceLoss(nn.Module):
    """Criterion class for computing combined BCE and Dice losses."""

    def __init__(self, weight_bce: float = 0.5, weight_dice: float = 0.5):
        """Initialize BCEDiceLoss with BCE and Dice weight factors.

        Args:
            weight_bce (float): Weight factor for BCE loss component.
            weight_dice (float): Weight factor for Dice loss component.
        """
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = MultiChannelDiceLoss(smooth=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate combined BCE and Dice loss between predictions and targets."""
        _, _, mask_h, mask_w = pred.shape
        if tuple(target.shape[-2:]) != (mask_h, mask_w):  # downsample to the same size as pred
            target = F.interpolate(target, (mask_h, mask_w), mode="nearest")
        return self.weight_bce * self.bce(pred, target) + self.weight_dice * self.dice(pred, target)


class KeypointLoss(nn.Module):
    """Criterion class for computing keypoint losses."""

    def __init__(self, sigmas: torch.Tensor) -> None:
        """Initialize the KeypointLoss class with keypoint sigmas."""
        super().__init__()
        self.sigmas = sigmas

    def forward(
        self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Calculate keypoint loss factor and Euclidean distance loss for keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses for YOLOv8 object detection."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):  # model must be de-paralleled
        """Initialize v8DetectionLoss with model parameters and task-aligned assignment settings."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        
        # Focal loss
        g = 1  # focal loss gamma
        if g > 0:
            self.bce = AdaptiveThresholdFocalLoss(self.bce, g)
            
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
            hyp=self.hyp,  # ★ 傳入 hyp
        )
        self.bbox_loss = BboxLoss(m.reg_max, hyp=self.hyp).to(device)  # ★ 傳入 hyp
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets by converting to tensor format and scaling coordinates."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).float().softmax(3).matmul(self.proj.type(torch.float))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size and return foreground mask and
        target indices.
        """
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, moe
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
                ((imgsz[0] ** 2 + imgsz[1] ** 2) / torch.square(stride_tensor)).repeat(1, batch_size).transpose(1, 0)
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        
        # MoE auxiliary loss
        moe_loss = torch.tensor(0.0, device=self.device)
        if hasattr(self, 'model'):
             for m in self.model.modules():
                 if hasattr(m, 'aux_loss'):
                     moe_loss += m.aux_loss
        loss[3] = moe_loss * self.hyp.moe
        
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )  # loss(box, cls, dfl, moe)

    def parse_output(
        self, preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Parse model predictions to extract features."""
        return preds[1] if isinstance(preds, tuple) else preds

    def __call__(
        self,
        preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """A wrapper for get_assigned_targets_and_loss and parse_output."""
        batch_size = preds["boxes"].shape[0]
        loss, loss_detach = self.get_assigned_targets_and_loss(preds, batch)[1:]
        return loss * batch_size, loss_detach
    
    
class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 segmentation."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):  # model must be de-paralleled
        """Initialize the v8SegmentationLoss class with model parameters and mask overlap setting."""
        super().__init__(model, tal_topk, tal_topk2)
        self.overlap = model.args.overlap_mask
        self.bcedice_loss = BCEDiceLoss(weight_bce=0.5, weight_dice=0.5)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the combined loss for detection and segmentation."""
        
        # 【修改點 1】：分開獲取 pred_masks 和 proto
        pred_masks = preds["mask_coefficient"].permute(0, 2, 1).contiguous()
        proto = preds["proto"]
        
        # 【修改點 2】：直接從 preds 字典中提取 semseg（對應 Segment 類別中的修改）
        pred_semantic = preds.get("semseg", None)
        
        # 初始化 6 個 loss: box, seg, cls, dfl, sem, moe
        loss = torch.zeros(6, device=self.device)  
        
        (fg_mask, target_gt_idx, target_bboxes, _, _), det_loss, _ = self.get_assigned_targets_and_loss(preds, batch)
        # NOTE: re-assign index for consistency for now. Need to be removed in the future.
        loss[0], loss[2], loss[3] = det_loss[0], det_loss[1], det_loss[2]

        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        if fg_mask.sum():
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                # masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]
                proto = F.interpolate(proto, masks.shape[-2:], mode="bilinear", align_corners=False)

            imgsz = (
                torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_masks.dtype) * self.stride[0]
            )
            loss[1] = self.calculate_segmentation_loss(
                fg_mask,
                masks,
                target_gt_idx,
                target_bboxes,
                batch["batch_idx"].view(-1, 1),
                proto,
                pred_masks,
                imgsz,
            )
            
            # 【保留】：計算語義分割損失 (Semantic Segmentation Loss)
            if pred_semantic is not None:
                sem_masks = batch["sem_masks"].to(self.device)  # NxHxW
                mask_zero = sem_masks == 0  # NxHxW
                sem_masks = F.one_hot(sem_masks.long(), num_classes=self.nc).permute(0, 3, 1, 2).float()  # NxCxHxW
                sem_masks[mask_zero.unsqueeze(1).expand_as(sem_masks)] = 0
                loss[4] = self.bcedice_loss(pred_semantic, sem_masks)
                loss[4] *= self.hyp.box  # seg gain

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss
            if pred_semantic is not None:
                loss[4] += (pred_semantic * 0).sum()

        loss[1] *= self.hyp.box  # seg gain

        # MoE auxiliary loss
        moe_loss = torch.tensor(0.0, device=self.device)
        if hasattr(self, 'model'):
             for m in self.model.modules():
                 if hasattr(m, 'aux_loss'):
                     moe_loss += m.aux_loss
        loss[5] = moe_loss * self.hyp.moe

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (N, H, W), where N is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (N, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (N, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (N,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if self.overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


# class v8SegmentationLoss(v8DetectionLoss):
#     """Criterion class for computing training losses for YOLOv8 segmentation."""

#     def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):  # model must be de-paralleled
#         """Initialize the v8SegmentationLoss class with model parameters and mask overlap setting."""
#         super().__init__(model, tal_topk, tal_topk2)
#         self.overlap = model.args.overlap_mask
#         self.bcedice_loss = BCEDiceLoss(weight_bce=0.5, weight_dice=0.5)

#     def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
#         """Calculate and return the combined loss for detection and segmentation."""
#         pred_masks, proto = preds["mask_coefficient"].permute(0, 2, 1).contiguous(), preds["proto"]
#         loss = torch.zeros(6, device=self.device)  # box, seg, cls, dfl, moe
#         if isinstance(proto, tuple) and len(proto) == 2:
#             proto, pred_semseg = proto
#         else:
#             pred_semseg = None
#         (fg_mask, target_gt_idx, target_bboxes, _, _), det_loss, _ = self.get_assigned_targets_and_loss(preds, batch)
#         # NOTE: re-assign index for consistency for now. Need to be removed in the future.
#         loss[0], loss[2], loss[3] = det_loss[0], det_loss[1], det_loss[2]

#         batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
#         if fg_mask.sum():
#             # Masks loss
#             masks = batch["masks"].to(self.device).float()
#             if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
#                 # masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]
#                 proto = F.interpolate(proto, masks.shape[-2:], mode="bilinear", align_corners=False)

#             imgsz = (
#                 torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_masks.dtype) * self.stride[0]
#             )
#             loss[1] = self.calculate_segmentation_loss(
#                 fg_mask,
#                 masks,
#                 target_gt_idx,
#                 target_bboxes,
#                 batch["batch_idx"].view(-1, 1),
#                 proto,
#                 pred_masks,
#                 imgsz,
#             )
#             if pred_semseg is not None:
#                 sem_masks = batch["sem_masks"].to(self.device)  # NxHxW
#                 mask_zero = sem_masks == 0  # NxHxW
#                 sem_masks = F.one_hot(sem_masks.long(), num_classes=self.nc).permute(0, 3, 1, 2).float()  # NxCxHxW
#                 sem_masks[mask_zero.unsqueeze(1).expand_as(sem_masks)] = 0
#                 loss[4] = self.bcedice_loss(pred_semseg, sem_masks)
#                 loss[4] *= self.hyp.box  # seg gain

#         # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
#         else:
#             loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss
#             if pred_semseg is not None:
#                 loss[4] += (pred_semseg * 0).sum()

#         loss[1] *= self.hyp.box  # seg gain

#         # MoE auxiliary loss
#         moe_loss = torch.tensor(0.0, device=self.device)
#         if hasattr(self, 'model'):
#              for m in self.model.modules():
#                  if hasattr(m, 'aux_loss'):
#                      moe_loss += m.aux_loss
#         loss[5] = moe_loss * self.hyp.moe

#         return loss * batch_size, loss.detach()  # loss(box, cls, dfl)

#     @staticmethod
#     def single_mask_loss(
#         gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
#     ) -> torch.Tensor:
#         """Compute the instance segmentation loss for a single image.

#         Args:
#             gt_mask (torch.Tensor): Ground truth mask of shape (N, H, W), where N is the number of objects.
#             pred (torch.Tensor): Predicted mask coefficients of shape (N, 32).
#             proto (torch.Tensor): Prototype masks of shape (32, H, W).
#             xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (N, 4).
#             area (torch.Tensor): Area of each ground truth bounding box of shape (N,).

#         Returns:
#             (torch.Tensor): The calculated mask loss for a single image.

#         Notes:
#             The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
#             predicted masks from the prototype masks and predicted mask coefficients.
#         """
#         pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
#         loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
#         return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

#     def calculate_segmentation_loss(
#         self,
#         fg_mask: torch.Tensor,
#         masks: torch.Tensor,
#         target_gt_idx: torch.Tensor,
#         target_bboxes: torch.Tensor,
#         batch_idx: torch.Tensor,
#         proto: torch.Tensor,
#         pred_masks: torch.Tensor,
#         imgsz: torch.Tensor,
#     ) -> torch.Tensor:
#         """Calculate the loss for instance segmentation.

#         Args:
#             fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
#             masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
#             target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
#             target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
#             batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
#             proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
#             pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
#             imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).

#         Returns:
#             (torch.Tensor): The calculated loss for instance segmentation.

#         Notes:
#             The batch loss can be computed for improved speed at higher memory usage.
#             For example, pred_mask can be computed as follows:
#                 pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
#         """
#         _, _, mask_h, mask_w = proto.shape
#         loss = 0

#         # Normalize to 0-1
#         target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

#         # Areas of target bboxes
#         marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

#         # Normalize to mask size
#         mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

#         for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
#             fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
#             if fg_mask_i.any():
#                 mask_idx = target_gt_idx_i[fg_mask_i]
#                 if self.overlap:
#                     gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
#                     gt_mask = gt_mask.float()
#                 else:
#                     gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

#                 loss += self.single_mask_loss(
#                     gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
#                 )

#             # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
#             else:
#                 loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

#         return loss / fg_mask.sum()
    
    
# 0, 1, 2 Pose loss    
# class v8PoseLoss(v8DetectionLoss):
#     """Criterion class for computing training losses for YOLOv8 pose estimation."""

#     def __init__(self, model, tal_topk: int = 10, tal_topk2: int = 10, alpha: float = 0.5):  # 增加 alpha 參數
#         """Initialize v8PoseLoss with model parameters and keypoint-specific loss functions."""
#         super().__init__(model, tal_topk, tal_topk2)
#         self.kpt_shape = model.model[-1].kpt_shape
#         self.bce_pose = nn.BCEWithLogitsLoss()
#         is_pose = self.kpt_shape == [17, 3]
#         nkpt = self.kpt_shape[0]  # number of keypoints
#         sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
#         self.keypoint_loss = KeypointLoss(sigmas=sigmas)
#         self.alpha = alpha  # 保存 alpha (0 < alpha < 1) 用於遮擋點權重

#     def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
#         """Calculate the total loss and detach it for pose estimation."""
#         pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
#         loss = torch.zeros(6, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility, moe
#         (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
#             self.get_assigned_targets_and_loss(preds, batch)
#         )
#         # NOTE: re-assign index for consistency for now. Need to be removed in the future.
#         loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

#         batch_size = pred_kpts.shape[0]
#         imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]

#         # Pboxes
#         pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

#         # Bbox loss
#         if fg_mask.sum():
#             keypoints = batch["keypoints"].to(self.device).float().clone()
#             keypoints[..., 0] *= imgsz[1]
#             keypoints[..., 1] *= imgsz[0]

#             loss[1], loss[2] = self.calculate_keypoints_loss(
#                 fg_mask,
#                 target_gt_idx,
#                 keypoints,
#                 batch["batch_idx"].view(-1, 1),
#                 stride_tensor,
#                 target_bboxes,
#                 pred_kpts,
#             )

#         loss[1] *= self.hyp.pose  # pose gain
#         loss[2] *= self.hyp.kobj  # kobj gain

        # # MoE auxiliary loss
        # moe_loss = torch.tensor(0.0, device=self.device)
        # if hasattr(self, 'model'):
        #      for m in self.model.modules():
        #          if hasattr(m, 'aux_loss'):
        #              moe_loss += m.aux_loss
        # loss[5] = moe_loss * self.hyp.moe

#         return loss * batch_size, loss.detach()  # loss(box, pose, kobj, cls, dfl)

#     @staticmethod
#     def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
#         """Decode predicted keypoints to image coordinates."""
#         y = pred_kpts.clone()
#         y[..., :2] *= 2.0
#         y[..., 0] += anchor_points[:, [0]] - 0.5
#         y[..., 1] += anchor_points[:, [1]] - 0.5
#         return y

#     def _select_target_keypoints(
#         self,
#         keypoints: torch.Tensor,
#         batch_idx: torch.Tensor,
#         target_gt_idx: torch.Tensor,
#         masks: torch.Tensor,
#     ) -> torch.Tensor:
#         """Select target keypoints for each anchor based on batch index and target ground truth index."""
#         # 此函數保持原樣
#         batch_idx = batch_idx.flatten()
#         batch_size = len(masks)
#         max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()
#         batched_keypoints = torch.zeros(
#             (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
#         )
#         for i in range(batch_size):
#             keypoints_i = keypoints[batch_idx == i]
#             batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i
#         target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)
#         selected_keypoints = batched_keypoints.gather(
#             1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
#         )
#         return selected_keypoints

#     def calculate_keypoints_loss(
#         self,
#         masks: torch.Tensor,
#         target_gt_idx: torch.Tensor,
#         keypoints: torch.Tensor,
#         batch_idx: torch.Tensor,
#         stride_tensor: torch.Tensor,
#         target_bboxes: torch.Tensor,
#         pred_kpts: torch.Tensor,
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         """Calculate the keypoints loss for the model."""
#         # Select target keypoints using helper method
#         selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)

#         # Divide coordinates by stride
#         selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

#         kpts_loss = 0
#         kpts_obj_loss = 0

#         if masks.any():
#             target_bboxes /= stride_tensor
#             gt_kpt = selected_keypoints[masks]
#             area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
#             pred_kpt = pred_kpts[masks]

#             # --- 修改部分開始 ---
#             # 判斷是否包含 visibility 維度 (通常 shape 為 [..., 3])
#             if gt_kpt.shape[-1] == 3:
#                 # 獲取 visibility 標籤: 0=不存在, 1=遮擋, 2=可見
#                 visibility = gt_kpt[..., 2]
                
#                 # 根據公式 w(v) 構建權重張量
#                 # v=0 -> w=0
#                 # v=1 -> w=alpha
#                 # v=2 -> w=1
#                 kpt_weights = torch.zeros_like(visibility)
#                 kpt_weights[visibility == 2] = 1.0
#                 kpt_weights[visibility == 1] = self.alpha
#                 # v=0 保持為 0.0

#                 # 計算關鍵點回歸 Loss (坐標誤差)
#                 # 傳入 kpt_weights 而不是 binary mask，這樣遮擋點的坐標誤差會被乘以 alpha
#                 kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_weights, area)

#                 if pred_kpt.shape[-1] == 3:
#                     # 計算關鍵點可見度 Loss (Objectness)
#                     # Target: 只要 v > 0 (即 1 或 2)，我們都希望模型預測 "存在" (Target=1)
#                     # 但是，我們對 Loss 使用 kpt_weights 進行加權
#                     # 這樣模型對於遮擋點(v=1)的預測錯誤懲罰較小 (alpha)，對可見點(v=2)懲罰較大 (1.0)
#                     target = (visibility > 0).float()
#                     kpts_obj_loss = F.binary_cross_entropy_with_logits(
#                         pred_kpt[..., 2], target, weight=kpt_weights, reduction='mean'
#                     )
#             else:
#                 # 如果數據集只有 x,y 沒有 visibility (2D)，則假設全部可見且權重為 1
#                 kpt_mask = torch.full_like(gt_kpt[..., 0], True) # 權重全為 1
#                 kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)

#                 if pred_kpt.shape[-1] == 3:
#                     # 如果沒有 GT visibility，則所有點的 target 設為 1，權重設為 1
#                     # 這裡使用原有的逻辑
#                     kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())
#             # --- 修改部分結束 ---

#         return kpts_loss, kpts_obj_loss

# RTMO Pose loss
class RTMOPoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLO pose estimation with RTMO strategy."""
    def __init__(self, model, tal_topk: int = 10, tal_topk2: int = 10):
        super().__init__(model, tal_topk, tal_topk2)
        self.model = model  
        self.current_epoch = 0 
        self.kpt_shape = model.model[-1].kpt_shape
        self.pose_vec_channels = model.model[-1].pose_vec_channels
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0] 
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)
        self.mle_loss = MLECCLoss()
        self.bbox_padding = 1.25
        
        # Two-stage Training 設定 (可根據實際總 epoch 調整)
        self.stage2_epoch = int(self.hyp.epochs*7/15)  # 論文中 COCO 訓練前 280 epochs 屬第一階段   (280, 600)   7:15 

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        pred_kpts_features = preds["kpts"]            # [bs, out_channels, anchors] (DCC特徵)
        pred_proxy_kpts = preds["proxy_kpts"]         # [bs, K*2, anchors] (Proxy回歸)
        
        # 保持 7 個 loss:[box, kpt_proxy, kpt_visibility, cls, dfl, mle, moe]
        loss = torch.zeros(7, device=self.device)  
        
        bs, _, anchors = pred_proxy_kpts.shape
        K = self.kpt_shape[0]

        # --- 為了讓 SimOTA 正確運作，我們必須把 proxy 組裝成 YOLO 標準格式 [x, y, v, ...] ---
        assigner_kpts = torch.zeros((bs, K * (2 if self.kpt_shape[1]==2 else 3), anchors), device=self.device)
        assigner_kpts[:, 0::(2 if self.kpt_shape[1]==2 else 3), :] = pred_proxy_kpts[:, 0::2, :] # x
        assigner_kpts[:, 1::(2 if self.kpt_shape[1]==2 else 3), :] = pred_proxy_kpts[:, 1::2, :] # y
        if self.kpt_shape[1] == 3:
            # Visibility 使用 DCC 那邊的輸出
            assigner_kpts[:, 2::3, :] = pred_kpts_features[:, self.pose_vec_channels:, :] 
            
        preds_for_assigner = preds.copy()
        preds_for_assigner["kpts"] = assigner_kpts

        # 這裡將使用 proxy 分支進行 SimOTA 樣本分配
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds_for_assigner, batch)
        )
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_proxy_kpts.dtype) * self.stride[0]

        if fg_mask.sum():
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]
            
            # 從模型環境獲取當前 Epoch (若無則預設為 0，即永遠第一階段)
            # current_epoch = getattr(self.model, 'epoch', 0)

            loss[1], loss[2], loss[5] = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,
                target_bboxes,
                pred_kpts_features.permute(0, 2, 1).contiguous(),
                pred_proxy_kpts.permute(0, 2, 1).contiguous(),
                anchor_points,
                # self.current_epoch
            )

        # ==========================================
        # ★ 論文 Section 3.3 的 Loss 權重 ★
        # ==========================================
        if self.current_epoch < self.stage2_epoch:
            # loss[0] *= 5.0   # λ1 (Bbox IoU Loss) = 5
            loss[5] *= 0.01   # λ2 (MLE Loss on DCC) = 5
            loss[1] *= 30.0  # λ3 (Proxy OKS Loss) = 10
            # loss[3] *= 2.0   # λ4 (Cls Varifocal Loss) = 2
        else:
            loss[5] *= 5.0  # λ2 (MLE Loss on DCC) = 5
            loss[1] *= 10.0  # λ3 (Proxy OKS Loss) = 10
            loss[3] *= 2.0   # λ4 (Cls Varifocal Loss) = 2

        # loss[2] *= getattr(self.hyp, 'kobj', 1.0) # Vis loss 論文未提權重，維持預設
        loss[2] *= self.hyp.kobj  # kobj gain

        moe_loss = torch.tensor(0.0, device=self.device)
        if hasattr(self, 'model'):
             for m in self.model.modules():
                 if hasattr(m, 'aux_loss'):
                     moe_loss += m.aux_loss
        loss[6] = moe_loss * getattr(self.hyp, 'moe', 1.0)

        return loss.sum() * bs, loss.detach()

    def _select_target_keypoints(self, keypoints: torch.Tensor, batch_idx: torch.Tensor, target_gt_idx: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()
        batched_keypoints = torch.zeros((batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device)

        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )
        return selected_keypoints

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts_features: torch.Tensor,
        pred_proxy_kpts: torch.Tensor,
        anchor_points: torch.Tensor,
        # current_epoch_2: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)
        
        proxy_loss = torch.tensor(0.0, device=masks.device)
        kpts_obj_loss = torch.tensor(0.0, device=masks.device)
        mle_loss = torch.tensor(0.0, device=masks.device)

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            pred_kpt_feat = pred_kpts_features[masks]
            pred_proxy = pred_proxy_kpts[masks]

            pose_vecs = pred_kpt_feat[:, :self.pose_vec_channels]
            pred_vis = pred_kpt_feat[:, self.pose_vec_channels:] if self.kpt_shape[1] == 3 else None
            
            dtype = pose_vecs.dtype

            target_bboxes_pos = target_bboxes[masks]
            center = (target_bboxes_pos[:, :2] + target_bboxes_pos[:, 2:]) / 2.0
            scale = (target_bboxes_pos[:, 2:] - target_bboxes_pos[:, :2]) * self.bbox_padding
            bbox_cs = torch.cat([center, scale], dim=-1).to(dtype)

            batch_size = masks.shape[0]  # Image batch size
            
            # [修正] 展開 anchor 與 stride，以便用 shape 為[bs, anchors] 的 masks 進行安全索引
            # 展開後的 anchor_points_pos 與 stride_tensor_pos 的 shape 將會是 [N, 2] 和 [N, 1]
            grids = (anchor_points * stride_tensor).unsqueeze(0).expand(batch_size, -1, -1)[masks].to(dtype)
            anchor_points_pos = anchor_points.unsqueeze(0).expand(batch_size, -1, -1)[masks].to(dtype)
            stride_tensor_pos = stride_tensor.unsqueeze(0).expand(batch_size, -1, -1)[masks].to(dtype)

            # --- 1. 計算 MLE Loss (作用於 DCC 分支) ---
            kpt_cc_preds, pred_hms, sigmas = self.model.model[-1].dcc.forward_train(pose_vecs, bbox_cs, grids)
            real_area = (target_bboxes_pos[:, 2:] - target_bboxes_pos[:, :2]).prod(1, keepdim=True).to(dtype)
            target_hms = self.model.model[-1].dcc.generate_target_heatmap(gt_kpt[..., :2].to(dtype), bbox_cs, sigmas, real_area.squeeze(1))

            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpt_mask_dtype = kpt_mask.to(dtype)

            mle_loss = self.mle_loss(pred_hms, target_hms, kpt_mask_dtype)

            # --- 2. 計算 Proxy Loss (作用於輕量級回歸分支) ---
            # [修正] view 的第一個維度改成 -1 (自動推斷正樣本數 N)
            # 並套用修正後的 anchor_points_pos 和 stride_tensor_pos 來解碼
            pred_proxy_decoded = (pred_proxy.view(-1, self.kpt_shape[0], 2) * 2.0 - 0.5 + anchor_points_pos.unsqueeze(1)) * stride_tensor_pos.unsqueeze(1)
            
            # ★ 兩階段訓練切換 ★
            if self.current_epoch < self.stage2_epoch:
                # 第一階段：Proxy 學習真實標籤
                proxy_target = gt_kpt.to(dtype)
            else:
                # 第二階段：Proxy 學習 DCC 預測出的高精度坐標
                proxy_target = gt_kpt.clone().to(dtype)
                proxy_target[..., :2] = kpt_cc_preds.detach()

            proxy_loss = self.keypoint_loss(pred_proxy_decoded, proxy_target, kpt_mask_dtype, real_area)

            # --- 3. 計算 Vis Loss ---
            if pred_vis is not None:
                kpts_obj_loss = self.bce_pose(pred_vis, kpt_mask.float())

        return proxy_loss, kpts_obj_loss, mle_loss


# 原版 Pose loss
class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int = 10):  # model must be de-paralleled
        """Initialize v8PoseLoss with model parameters and keypoint-specific loss functions."""
        super().__init__(model, tal_topk, tal_topk2)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss and detach it for pose estimation."""
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
        loss = torch.zeros(6, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility, moe
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )
        # NOTE: re-assign index for consistency for now. Need to be removed in the future.
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        batch_size = pred_kpts.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]

        # Pboxes
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        # Bbox loss
        if fg_mask.sum():
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,
                target_bboxes,
                pred_kpts,
            )

        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain

        # MoE auxiliary loss
        moe_loss = torch.tensor(0.0, device=self.device)
        if hasattr(self, 'model'):
             for m in self.model.modules():
                 if hasattr(m, 'aux_loss'):
                     moe_loss += m.aux_loss
        loss[5] = moe_loss * self.hyp.moe

        return loss * batch_size, loss.detach()  # loss(box, pose, kobj, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def _select_target_keypoints(
        self,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        target_gt_idx: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Select target keypoints for each anchor based on batch index and target ground truth index.

        Args:
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).

        Returns:
            (torch.Tensor): Selected keypoints tensor, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        return selected_keypoints

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        # Select target keypoints using helper method
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            target_bboxes /= stride_tensor
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class RLEPoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation with Rle-Oks Loss support."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int = 10):  # model must be de-paralleled
        """Initialize v8PoseLoss with model parameters and keypoint-specific loss functions."""
        super().__init__(model, tal_topk, tal_topk2)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

        # --- Rle-Oks Loss Components ---
        self.flow_model = model.model[-1].flow_model if hasattr(model.model[-1], "flow_model") else None
        if self.flow_model is not None:
            self.rle_loss = RLELoss(use_target_weight=True).to(self.device)
            self.target_weights = (
                torch.from_numpy(RLE_WEIGHT).to(self.device) if is_pose else torch.ones(nkpt, device=self.device)
            )

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss and detach it for pose estimation."""
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
        has_rle = getattr(self, "rle_loss", None) is not None
        loss = torch.zeros(7 if has_rle else 6, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility, rle, moe
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        batch_size = pred_kpts.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]

        pred_kpts = pred_kpts.view(batch_size, -1, *self.kpt_shape)  # (b, h*w, 17, 3)

        if has_rle and preds.get("kpts_sigma", None) is not None:
            pred_sigma = preds["kpts_sigma"].permute(0, 2, 1).contiguous()
            pred_sigma = pred_sigma.view(batch_size, -1, self.kpt_shape[0], 2)  # (b, h*w, 17, 2)
            pred_kpts = torch.cat([pred_kpts, pred_sigma], dim=-1)  # (b, h*w, 17, 5)

        # Pboxes
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts)  

        # Bbox loss
        if fg_mask.sum():
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            keypoints_loss = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,
                target_bboxes,
                pred_kpts,
            )
            loss[1] = keypoints_loss[0]
            loss[2] = keypoints_loss[1]
            if has_rle:
                loss[5] = keypoints_loss[2]

        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain

        # MoE auxiliary loss
        moe_loss = torch.tensor(0.0, device=self.device)
        if hasattr(self, 'model'):
             for m in self.model.modules():
                 if hasattr(m, 'aux_loss'):
                     moe_loss += m.aux_loss
        loss[6] = moe_loss * self.hyp.moe

        return loss * batch_size, loss.detach()  # loss(box, pose, kobj, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., 0] += anchor_points[:, [0]]
        y[..., 1] += anchor_points[:, [1]]
        return y

    def calculate_rle_oks_loss(self, pred_kpt: torch.Tensor, gt_kpt: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor) -> torch.Tensor:
        """Calculate the Rle-Oks loss for keypoints, referencing OKS evaluation parameters."""
        pred_kpt_visible = pred_kpt[kpt_mask]
        gt_kpt_visible = gt_kpt[kpt_mask]

        if pred_kpt_visible.shape[0] == 0:
            return torch.tensor(0.0, device=pred_kpt.device)

        pred_coords = pred_kpt_visible[:, 0:2]
        pred_sigma = pred_kpt_visible[:, -2:].sigmoid()
        gt_coords = gt_kpt_visible[:, 0:2]

        # Targets weights distribution 
        target_weights = self.target_weights.unsqueeze(0).repeat(kpt_mask.shape[0], 1)[kpt_mask]

        # 展開 Area 和 Sigmas 來適應 Normalize
        area_expanded = area.unsqueeze(1).repeat(1, gt_kpt.shape[1], 1)
        area_visible = area_expanded[kpt_mask] # Shape (N_vis, 1)

        sigmas = self.keypoint_loss.sigmas
        sigmas_expanded = sigmas.unsqueeze(0).unsqueeze(-1).repeat(kpt_mask.shape[0], 1, 1)
        sigmas_visible = sigmas_expanded[kpt_mask] # Shape (N_vis, 1)

        # epsilon = 1 / (2 * sqrt(area) * k_i) -> 考慮 s 與 k_i
        epsilon = 1.0 / (2.0 * torch.sqrt(area_visible + 1e-9) * sigmas_visible + 1e-9)

        error = (pred_coords - gt_coords) / (pred_sigma + 1e-9)
        error_oks = error * epsilon

        # 防止流模型中的驗證崩潰 (NaN/Inf)
        valid_mask = ~(torch.isnan(error_oks) | torch.isinf(error_oks)).any(dim=-1)
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred_kpt.device)

        error_oks = error_oks[valid_mask].clamp(-100, 100)
        pred_sigma = pred_sigma[valid_mask]
        epsilon = epsilon[valid_mask]
        target_weights = target_weights[valid_mask]

        log_phi = self.flow_model.log_prob(error_oks)
        
        # 搭配新分佈 \log(\hat{\sigma} * \epsilon) 使用
        sigma_oks = pred_sigma * epsilon

        return self.rle_loss(sigma_oks, log_phi, error_oks, target_weights)

    def _select_target_keypoints(
        self,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        target_gt_idx: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Select target keypoints for each anchor based on batch index and target ground truth index.

        Args:
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).

        Returns:
            (torch.Tensor): Selected keypoints tensor, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        return selected_keypoints

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        # Select target keypoints using helper method
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0
        rle_loss = 0

        if masks.any():
            target_bboxes /= stride_tensor
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            # Rle-Oks Loss 判斷
            if getattr(self, "rle_loss", None) is not None and (pred_kpt.shape[-1] == 4 or pred_kpt.shape[-1] == 5):
                rle_loss = self.calculate_rle_oks_loss(pred_kpt, gt_kpt, kpt_mask, area)

            if pred_kpt.shape[-1] == 3 or pred_kpt.shape[-1] == 5:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss, rle_loss


class PoseLoss26(v8PoseLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation with RLE loss support."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):  # model must be de-paralleled
        """Initialize PoseLoss26 with model parameters and keypoint-specific loss functions including RLE loss."""
        super().__init__(model, tal_topk, tal_topk2)
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        self.rle_loss = None
        self.flow_model = model.model[-1].flow_model if hasattr(model.model[-1], "flow_model") else None
        if self.flow_model is not None:
            self.rle_loss = RLELoss(use_target_weight=True).to(self.device)
            self.target_weights = (
                torch.from_numpy(RLE_WEIGHT).to(self.device) if is_pose else torch.ones(nkpt, device=self.device)
            )

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss and detach it for pose estimation."""
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
        loss = torch.zeros(7 if self.rle_loss else 5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility, moe
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )
        # NOTE: re-assign index for consistency for now. Need to be removed in the future.
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        batch_size = pred_kpts.shape[0]
        imgsz = torch.tensor(batch["resized_shape"][0], device=self.device, dtype=pred_kpts.dtype)  # image size (h,w)

        pred_kpts = pred_kpts.view(batch_size, -1, *self.kpt_shape)  # (b, h*w, 17, 3)

        if self.rle_loss and preds.get("kpts_sigma", None) is not None:
            pred_sigma = preds["kpts_sigma"].permute(0, 2, 1).contiguous()
            pred_sigma = pred_sigma.view(batch_size, -1, self.kpt_shape[0], 2)  # (b, h*w, 17, 2)
            pred_kpts = torch.cat([pred_kpts, pred_sigma], dim=-1)  # (b, h*w, 17, 5)

        pred_kpts = self.kpts_decode(anchor_points, pred_kpts)

        # Bbox loss
        if fg_mask.sum():
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            keypoints_loss = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,
                target_bboxes,
                pred_kpts,
            )
            loss[1] = keypoints_loss[0]
            loss[2] = keypoints_loss[1]
            if self.rle_loss is not None:
                loss[5] = keypoints_loss[2]

        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        if self.rle_loss is not None:
            loss[5] *= self.hyp.rle  # rle gain

        # MoE auxiliary loss
        moe_loss = torch.tensor(0.0, device=self.device)
        if hasattr(self, 'model'):
             for m in self.model.modules():
                 if hasattr(m, 'aux_loss'):
                     moe_loss += m.aux_loss
        loss[6] = moe_loss * self.hyp.moe

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl, kpt_location, kpt_visibility)

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., 0] += anchor_points[:, [0]]
        y[..., 1] += anchor_points[:, [1]]
        return y

    def calculate_rle_loss(self, pred_kpt: torch.Tensor, gt_kpt: torch.Tensor, kpt_mask: torch.Tensor) -> torch.Tensor:
        """Calculate the RLE (Residual Log-likelihood Estimation) loss for keypoints.

        Args:
            pred_kpt (torch.Tensor): Predicted keypoints with sigma, shape (N, kpts_dim) where kpts_dim >= 4.
            gt_kpt (torch.Tensor): Ground truth keypoints, shape (N, kpts_dim).
            kpt_mask (torch.Tensor): Mask for valid keypoints, shape (N, num_keypoints).

        Returns:
            (torch.Tensor): The RLE loss.
        """
        pred_kpt_visible = pred_kpt[kpt_mask]
        gt_kpt_visible = gt_kpt[kpt_mask]
        pred_coords = pred_kpt_visible[:, 0:2]
        pred_sigma = pred_kpt_visible[:, -2:]
        gt_coords = gt_kpt_visible[:, 0:2]

        target_weights = self.target_weights.unsqueeze(0).repeat(kpt_mask.shape[0], 1)
        target_weights = target_weights[kpt_mask]

        pred_sigma = pred_sigma.sigmoid()
        error = (pred_coords - gt_coords) / (pred_sigma + 1e-9)

        # Filter out NaN and Inf values to prevent MultivariateNormal validation errors
        valid_mask = ~(torch.isnan(error) | torch.isinf(error)).any(dim=-1)
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred_kpt.device)

        error = error[valid_mask]
        error = error.clamp(-100, 100)  # Prevent numerical instability
        pred_sigma = pred_sigma[valid_mask]
        target_weights = target_weights[valid_mask]

        log_phi = self.flow_model.log_prob(error)

        return self.rle_loss(pred_sigma, log_phi, error, target_weights)

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
            rle_loss (torch.Tensor): The RLE loss.
        """
        # Select target keypoints using inherited helper method
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0
        rle_loss = 0

        if masks.any():
            target_bboxes /= stride_tensor
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if self.rle_loss is not None and (pred_kpt.shape[-1] == 4 or pred_kpt.shape[-1] == 5):
                rle_loss = self.calculate_rle_loss(pred_kpt, gt_kpt, kpt_mask)
            if pred_kpt.shape[-1] == 3 or pred_kpt.shape[-1] == 5:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss, rle_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses for classification."""

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the classification loss between predictions and true labels."""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        return loss, loss.detach()


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model, tal_topk=10, tal_topk2: int | None = None):
        """Initialize v8OBBLoss with model, assigner, and rotated bbox loss; model must be de-paralleled."""
        super().__init__(model, tal_topk=tal_topk)
        self.assigner = RotatedTaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets for oriented bounding box detection."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the loss for oriented bounding box detection."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, angle, moe
        pred_distri, pred_scores, pred_angle = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
            preds["angle"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width

        dtype = pred_scores.dtype
        imgsz = torch.tensor(batch["resized_shape"][0], device=self.device, dtype=dtype)  # image size (h,w)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * float(imgsz[1]), targets[:, 5] * float(imgsz[0])
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo26n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )
            weight = target_scores.sum(-1)[fg_mask]
            loss[3] = self.calculate_angle_loss(
                pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum
            )  # angle loss
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        loss[3] *= self.hyp.angle  # angle gain

         # MoE auxiliary loss
        moe_loss = torch.tensor(0.0, device=self.device)
        if hasattr(self, 'model'):
             for m in self.model.modules():
                 if hasattr(m, 'aux_loss'):
                     moe_loss += m.aux_loss
        loss[4] = moe_loss * self.hyp.moe

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl, angle)

    def bbox_decode(
        self, anchor_points: torch.Tensor, pred_dist: torch.Tensor, pred_angle: torch.Tensor
    ) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)

    def calculate_angle_loss(self, pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum, lambda_val=3):
        """Calculate oriented angle loss.

        Args:
            pred_bboxes: [N, 5] (x, y, w, h, theta).
            target_bboxes: [N, 5] (x, y, w, h, theta).
            fg_mask: Foreground mask indicating valid predictions.
            weight: Loss weights for each prediction.
            target_scores_sum: Sum of target scores for normalization.
            lambda_val: control the sensitivity to aspect ratio.
        """
        w_gt = target_bboxes[..., 2]
        h_gt = target_bboxes[..., 3]
        pred_theta = pred_bboxes[..., 4]
        target_theta = target_bboxes[..., 4]

        log_ar = torch.log(w_gt / h_gt)
        scale_weight = torch.exp(-(log_ar**2) / (lambda_val**2))

        delta_theta = pred_theta - target_theta
        delta_theta_wrapped = delta_theta - torch.round(delta_theta / math.pi) * math.pi
        ang_loss = torch.sin(2 * delta_theta_wrapped[fg_mask]) ** 2

        ang_loss = scale_weight[fg_mask] * ang_loss
        ang_loss = ang_loss * weight

        return ang_loss.sum() / target_scores_sum


class E2EDetectLoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]


class E2ELoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model, loss_fn=v8DetectionLoss):
        """Initialize E2ELoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = loss_fn(model, tal_topk=10)
        self.one2one = loss_fn(model, tal_topk=7, tal_topk2=1)
        self.updates = 0
        self.total = 1.0
        # init gain
        self.o2m = 0.8
        self.o2o = self.total - self.o2m
        self.o2m_copy = self.o2m
        # final gain
        self.final_o2m = 0.1

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = self.one2many.parse_output(preds)
        one2many, one2one = preds["one2many"], preds["one2one"]
        loss_one2many = self.one2many.loss(one2many, batch)
        loss_one2one = self.one2one.loss(one2one, batch)
        return loss_one2many[0] * self.o2m + loss_one2one[0] * self.o2o, loss_one2one[1]

    def update(self) -> None:
        """Update the weights for one-to-many and one-to-one losses based on the decay schedule."""
        self.updates += 1
        self.o2m = self.decay(self.updates)
        self.o2o = max(self.total - self.o2m, 0)

    def decay(self, x) -> float:
        """Calculate the decayed weight for one-to-many loss based on the current update step."""
        return max(1 - x / max(self.one2one.hyp.epochs - 1, 1), 0) * (self.o2m_copy - self.final_o2m) + self.final_o2m


class TVPDetectLoss:
    """Criterion class for computing training losses for text-visual prompt detection."""

    def __init__(self, model, tal_topk=10, tal_topk2: int | None = None):
        """Initialize TVPDetectLoss with task-prompt and visual-prompt criteria using the provided model."""
        self.vp_criterion = v8DetectionLoss(model, tal_topk)
        # NOTE: store following info as it's changeable in __call__
        self.hyp = self.vp_criterion.hyp
        self.ori_nc = self.vp_criterion.nc
        self.ori_no = self.vp_criterion.no
        self.ori_reg_max = self.vp_criterion.reg_max

    def parse_output(self, preds) -> dict[str, torch.Tensor]:
        """Parse model predictions to extract features."""
        return self.vp_criterion.parse_output(preds)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""
        assert self.ori_reg_max == self.vp_criterion.reg_max  # TODO: remove it

        if self.ori_nc == preds["scores"].shape[1]:
            loss = torch.zeros(3, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        preds["scores"] = self._get_vp_features(preds)
        vp_loss = self.vp_criterion(preds, batch)
        box_loss = vp_loss[0][1]
        return box_loss, vp_loss[1]

    def _get_vp_features(self, preds: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """Extract visual-prompt features from the model output."""
        # NOTE: remove empty placeholder
        scores = preds["scores"][:, self.ori_nc :, :]
        vnc = scores.shape[1]

        self.vp_criterion.nc = vnc
        self.vp_criterion.no = vnc + self.vp_criterion.reg_max * 4
        self.vp_criterion.assigner.num_classes = vnc
        return scores


class TVPSegmentLoss(TVPDetectLoss):
    """Criterion class for computing training losses for text-visual prompt segmentation."""

    def __init__(self, model, tal_topk=10):
        """Initialize TVPSegmentLoss with task-prompt and visual-prompt criteria using the provided model."""
        super().__init__(model)
        self.vp_criterion = v8SegmentationLoss(model, tal_topk)
        self.hyp = self.vp_criterion.hyp

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt segmentation."""
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""
        assert self.ori_reg_max == self.vp_criterion.reg_max  # TODO: remove it

        if self.ori_nc == preds["scores"].shape[1]:
            loss = torch.zeros(4, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        preds["scores"] = self._get_vp_features(preds)
        vp_loss = self.vp_criterion(preds, batch)
        cls_loss = vp_loss[0][2]
        return cls_loss, vp_loss[1]
    
    
class SemanticSegmentationLoss(nn.Module):
    """Loss function for semantic segmentation using cross-entropy and Dice terms.

    Attributes:
        nc (int): Number of semantic classes.
        ce (nn.CrossEntropyLoss): Cross-entropy loss with ignore_index=255.
    """

    def __init__(self, model):
        """Initialize semantic segmentation loss.

        Args:
            model (torch.nn.Module): Model containing the SemanticSegment head.
        """
        super().__init__()
        m = model.model[-1]
        self.nc = m.nc
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        data_name = Path(str(getattr(model.args, "data", "") or "")).stem.lower()
        self.use_cityscapes_weight = data_name in {"cityscapes", "cityscapes8"} and self.nc == len(CITYSCAPES_WEIGHT)
        if self.nc == 1:
            self.ce = nn.BCEWithLogitsLoss()
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=255).to(device=self.device, dtype=self.dtype)
            if self.use_cityscapes_weight:
                # Non-persistent: weight is a deterministic constant, no need to serialize into ckpt state_dict.
                weight = torch.from_numpy(CITYSCAPES_WEIGHT).to(device=self.device, dtype=self.dtype)
                self.ce.register_buffer("weight", weight, persistent=False)

    def _resize_masks(self, masks, target_shape):
        """Resize masks to match prediction spatial dimensions."""
        if masks.shape[1:] != target_shape:
            return (
                F.interpolate(masks.float().unsqueeze(1), size=target_shape, mode="nearest").squeeze(1).to(torch.int32)
            )
        return masks

    def _ce_loss(self, preds, masks):
        """Compute cross-entropy on flattened pixels to avoid the CUDA nll_loss2d path."""
        if self.nc == 1:
            flat = masks.reshape(-1)
            valid = flat != 255
            logits = preds.reshape(-1)[valid]
            target = flat[valid].float()
        else:
            logits = preds.permute(0, 2, 3, 1).reshape(-1, self.nc)
            target = masks.reshape(-1).long()
        return self.ce(logits, target)

    def _dice_loss(self, preds, masks):
        """Compute Dice loss excluding ignore pixels."""
        if self.nc == 1:
            return self._binary_dice_loss(preds, masks)
        flat_target = masks.reshape(-1)
        valid = flat_target != 255
        if not valid.any():
            return preds.sum() * 0

        pred_soft = F.softmax(preds, dim=1)
        target = flat_target[valid].long()
        flat_pred = pred_soft.permute(0, 2, 3, 1).reshape(-1, self.nc)[valid]
        intersection = torch.zeros(self.nc, device=preds.device, dtype=pred_soft.dtype)
        intersection.scatter_add_(0, target, flat_pred.gather(1, target[:, None]).squeeze(1))
        pred_sum = flat_pred.sum(dim=0)
        target_sum = torch.bincount(target, minlength=self.nc).to(device=preds.device, dtype=pred_soft.dtype)
        cardinality = pred_sum + target_sum
        return (1.0 - (2.0 * intersection + 1.0) / (cardinality + 1.0)).mean()

    def _binary_dice_loss(self, preds, masks):
        """Compute Dice loss for single-class (binary) segmentation.

        Pixels with value 255 are excluded from Dice terms to match BCE valid-pixel filtering.
        """
        valid = (masks != 255).float()
        pred_soft = preds.squeeze(1).sigmoid()
        target = (masks == 1).float()
        intersection = (pred_soft * target * valid).sum()
        cardinality = ((pred_soft + target) * valid).sum()
        return 1.0 - (2.0 * intersection + 1.0) / (cardinality + 1.0)

    def forward(self, preds, batch):
        """Compute semantic segmentation loss with optional auxiliary loss.

        Args:
            preds (torch.Tensor | tuple): Main logits [B, nc, H', W'], or (main, aux) tuple.
            batch (dict): Batch dict with 'semantic_mask' [B, H, W] containing class IDs (255=ignore).

        Returns:
            (tuple[torch.Tensor, torch.Tensor]): (total_loss * batch_size, detached loss items [ce, dice, aux]).
        """
        # Unpack auxiliary logits when present.
        aux_logits = None
        if isinstance(preds, tuple):
            preds, aux_logits = preds

        masks = batch["semantic_mask"].to(preds.device)
        if preds.shape[2:] != masks.shape[1:]:
            preds = F.interpolate(preds, size=masks.shape[1:], mode="bilinear", align_corners=False)

        # Main cross-entropy and Dice loss.
        ce_loss = self._ce_loss(preds, masks)
        dice_loss = self._dice_loss(preds, masks)
        total = ce_loss + dice_loss

        # Auxiliary cross-entropy loss.
        aux_loss = torch.tensor(0.0, device=preds.device)
        if aux_logits is not None:
            if aux_logits.shape[2:] != masks.shape[1:]:
                aux_logits = F.interpolate(aux_logits, size=masks.shape[1:], mode="bilinear", align_corners=False)
            aux_loss = (self._ce_loss(aux_logits, masks)) * 0.4
            total = total + aux_loss

        loss_items = torch.stack([ce_loss, dice_loss, aux_loss]).detach()
        return total * preds.shape[0], loss_items
