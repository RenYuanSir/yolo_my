# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist
from typing import Tuple, Dict, List

# 常量：表示无意义的h_rel值
UNKNOWN_H = -1.0
LOGGER = logging.getLogger(__name__)

class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
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
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model, tal_topk=10):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
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
            out[..., 1:5] = xywh2xyxy(out[..., 1:5])
        return out



    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        
        # 检查是否是Detect_Efficient_Tomato头部输出的特殊格式(字典格式)
        has_h_pos = False
        pred_h = None
        
        # 处理不同格式的输入
        if isinstance(feats, dict):
            # 新格式：字典格式 {"features": [...], "h_pos": [...]}
            if "h_pos" in feats:
                has_h_pos = True
                pred_h = feats["h_pos"]
            
            if "features" in feats:
                feats = feats["features"]
        elif isinstance(feats, list) and len(feats) > 0 and isinstance(feats[-1], dict):
            # 旧格式：嵌套格式 [features, {"h_pos": [...]}]
            if "h_pos" in feats[-1]:
                has_h_pos = True
                pred_h = feats[-1]["h_pos"]
                feats = feats[0] if isinstance(feats[0], (list, tuple)) else feats[:-1]
        
        # 确保feats是一个可迭代且非嵌套的列表
        if isinstance(feats, list) and len(feats) > 0 and isinstance(feats[0], list):
            # 如果feats是列表的列表，取第一个子列表
            print(f"警告：feats是嵌套列表，取第一个子列表")
            feats = feats[0]
        
        # 调整连接维度，确保可以正确拆分
        try:
            # 诊断特征维度
            if not all(isinstance(xi, torch.Tensor) for xi in feats):
                raise ValueError(f"feats中包含非张量元素: {[type(xi) for xi in feats]}")
                
            shapes = [xi.shape for xi in feats]
            batch_size = feats[0].shape[0]
            no = self.reg_max * 4 + self.nc  # 输出通道数 = bbox分布 + 类别
            
            # 检查每个特征层的通道数是否符合预期
            expected_channels = [no for _ in range(len(feats))]
            actual_channels = [xi.shape[1] for xi in feats]
            
            # 如果通道数不一致，需要调整提取方式
            if not all(a == e for a, e in zip(actual_channels, expected_channels)):
                print(f"特征通道数不一致: 预期{expected_channels}, 实际{actual_channels}")
                # 分别提取bbox和cls分支
                bbox_feats = [xi[:, :self.reg_max * 4, ...] for xi in feats]
                cls_feats = [xi[:, self.reg_max * 4:(self.reg_max * 4 + self.nc), ...] for xi in feats]
                
                # 分别处理
                pred_distri = torch.cat([xi.view(batch_size, self.reg_max * 4, -1) for xi in bbox_feats], 2)
                pred_scores = torch.cat([xi.view(batch_size, self.nc, -1) for xi in cls_feats], 2)
            else:
                # 原始处理方式
                pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
                    (self.reg_max * 4, self.nc), 1
                )
        except (RuntimeError, ValueError, AttributeError) as e:
            # 如果出现错误，使用更稳健的特征处理方式
            print(f"处理特征时出错: {e}")
            print(f"特征类型: {type(feats)}")
            if isinstance(feats, list):
                print(f"特征元素类型: {[type(x) for x in feats]}")
            
            # 应急措施：如果feats是嵌套结构，尝试展平
            if isinstance(feats, list) and len(feats) > 0:
                if all(isinstance(x, torch.Tensor) for x in feats):
                    flat_feats = feats
                elif all(isinstance(x, list) for x in feats) and all(isinstance(y, torch.Tensor) for x in feats for y in x):
                    flat_feats = [tensor for sublist in feats for tensor in sublist]
                    print(f"展平后的特征长度: {len(flat_feats)}")
                else:
                    raise TypeError(f"无法处理的特征类型: {[type(x) for x in feats]}")
            else:
                raise TypeError(f"feats不是列表类型: {type(feats)}")
                
            # 分别处理每个特征层，确保维度一致
            batch_size = flat_feats[0].shape[0]
            bbox_feats = []
            cls_feats = []
            
            for xi in flat_feats:
                # 确保通道维度至少有足够的维度用于分割
                if xi.shape[1] >= (self.reg_max * 4 + self.nc):
                    bbox_feats.append(xi[:, :self.reg_max * 4, ...])
                    cls_feats.append(xi[:, self.reg_max * 4:(self.reg_max * 4 + self.nc), ...])
                else:
                    # 如果通道数不足，说明特征层可能有问题
                    print(f"警告: 特征层通道数不足 {xi.shape}")
                    # 使用零填充来确保维度兼容
                    padding_bbox = torch.zeros((batch_size, self.reg_max * 4, *xi.shape[2:]), device=xi.device)
                    padding_cls = torch.zeros((batch_size, self.nc, *xi.shape[2:]), device=xi.device)
                    bbox_feats.append(padding_bbox)
                    cls_feats.append(padding_cls)
            
            # 将各特征层拼接
            pred_distri = torch.cat([xi.view(batch_size, self.reg_max * 4, -1) for xi in bbox_feats], 2)
            pred_scores = torch.cat([xi.view(batch_size, self.nc, -1) for xi in cls_feats], 2)

        # 转换维度排列
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        
        # 处理h_pos预测，如果存在
        if has_h_pos and pred_h is not None:
            if isinstance(pred_h, list):
                # 将多层特征图的h_pos预测拼接起来
                try:
                    pred_h = torch.cat([xi.view(batch_size, 1, -1) for xi in pred_h], 2)
                    pred_h = pred_h.permute(0, 2, 1).contiguous()
                except Exception as e:
                    print(f"处理h_pos时出错: {e}")
                    has_h_pos = False  # 处理失败则禁用h_pos
            else:
                # 如果只有一个h_pos预测，确保其维度正确
                try:
                    pred_h = pred_h.permute(0, 2, 1).contiguous() if hasattr(pred_h, 'permute') else pred_h
                except Exception as e:
                    print(f"处理h_pos维度时出错: {e}")
                    has_h_pos = False  # 处理失败则禁用h_pos

        # 计算参数
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        cls_loss = self.bce(pred_scores, target_scores.to(dtype))  # BCE(pred_scores, target_scores)
        loss[1] = cls_loss.sum() / target_scores_sum

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

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
        cls_loss = self.bce(pred_scores, target_scores.to(dtype))  # BCE(pred_scores, target_scores)
        loss[2] = cls_loss.sum() / target_scores_sum

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

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
        overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

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
                if overlap:
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


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

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
        cls_loss = self.bce(pred_scores, target_scores.to(dtype))  # BCE(pred_scores, target_scores)
        loss[3] = cls_loss.sum() / target_scores_sum

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        """
        Calculate the keypoints loss for the model.

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

        # Divide coordinates by stride
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model):
        """Initializes v8OBBLoss with model, assigner, and rotated bbox loss; note model must be de-paralleled."""
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
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

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
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
        cls_loss = self.bce(pred_scores, target_scores.to(dtype))  # BCE(pred_scores, target_scores)
        loss[1] = cls_loss.sum() / target_scores_sum

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

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


class SafeBboxLoss:
    """
    安全版边界框损失，处理NaN和无效框
    """
    def __init__(self, reg_max=16):
        """初始化安全版边界框损失计算类"""
        self.reg_max = reg_max
        self.use_dfl = reg_max > 1
        self.device = None
        
    def to(self, device):
        """将实例移动到指定设备"""
        self.device = device
        return self
        
    def __call__(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, eps=1e-8):
        """
        计算边界框损失和DFL损失
        
        Args:
            pred_dist (tensor): 预测的分布
            pred_bboxes (tensor): 预测的边界框
            anchor_points (tensor): 锚点
            target_bboxes (tensor): 目标边界框
            target_scores (tensor): 目标分数
            target_scores_sum (float): 目标分数的总和
            fg_mask (tensor): 前景掩码
            eps (float): 防止除零的小常数
            
        Returns:
            tuple: IoU损失和DFL损失
        """
        # 处理空前景掩码情况
        if fg_mask.sum() == 0:
            return torch.tensor(0.0).to(pred_dist.device), torch.tensor(0.0).to(pred_dist.device)
              
        # 在前景位置提取预测和目标
        pred_bboxes_pos = pred_bboxes[fg_mask]
        pred_dist_pos = pred_dist[fg_mask]
        target_bboxes_pos = target_bboxes[fg_mask] 
        # scale target_bboxes from [0,1] → [0, image_size]

        target_scores_pos = target_scores[fg_mask]
        # 正确方式：target_scores_pos 是前面用 fg_mask 筛选后的
        weight = target_scores_pos.sum(-1).unsqueeze(-1)  # [N, 1]

        # ✅ 正确：target_scores_sum 是一个数
        target_scores_sum = target_scores_pos.sum()  # ✅ 保证是标量



        # 展平前景掩码，用于选择正样本位置
        B, N = fg_mask.shape  # 例如 [2, 9261]

        # 扩展 anchor_points 到 [B, N, 2] 以匹配 fg_mask
        anchor_points_expand = anchor_points.unsqueeze(0).expand(B, -1, -1)  # [B, N, 2]

        # 选择前景样本对应的 anchor points
        anchor_points_pos = anchor_points_expand[fg_mask]  # [num_fg, 2]

        
        # 检查并处理可能的NaN值
        if torch.isnan(target_bboxes_pos).any():
            LOGGER.warning("SafeBboxLoss: target_bboxes包含NaN，尝试修复!")
            target_bboxes_pos = torch.nan_to_num(target_bboxes_pos, 0.0)
            
        if torch.isnan(pred_bboxes_pos).any() or torch.isnan(pred_dist_pos).any():
            LOGGER.warning("SafeBboxLoss: pred_bboxes或pred_dist包含NaN!")
            pred_bboxes_pos = torch.nan_to_num(pred_bboxes_pos, 0.0)
            pred_dist_pos = torch.nan_to_num(pred_dist_pos, 0.0)
        
        # 计算IoU损失
        weight = target_scores_pos.sum(-1).unsqueeze(-1)
        iou = bbox_iou(pred_bboxes_pos, target_bboxes_pos, xywh=False, CIoU=True)
        iou = torch.clamp(iou, min=0.0, max=1.0)
        print(f"[DEBUG] iou.shape: {iou.shape}")
        print(f"[DEBUG] weight.shape: {weight.shape}")
        print(f"[DEBUG] target_scores_sum: {target_scores_sum}")
        print(f"[DEBUG] pred_dist_pos.shape = {pred_dist_pos.shape}")
        print(f"[DEBUG] target_scores_pos.sum(dim=-1): {target_scores_pos.sum(dim=-1)}")
        print(f"[DEBUG] target_scores.max(): {target_scores.max()}")
        print(f"[DEBUG] target_scores.min(): {target_scores.min()}")
        print(f"[DEBUG] anchor_points range: {anchor_points.min()}, {anchor_points.max()}")
        print(f"[DEBUG] pred_dist.min(): {pred_dist.min().item()}, max: {pred_dist.max().item()}")
        print(f"[DEBUG] pred_dist_pos stats: min={pred_dist_pos.min().item()}, max={pred_dist_pos.max().item()}, mean={pred_dist_pos.mean().item()}, std={pred_dist_pos.std().item()}")
        print(f"[DEBUG] fg_mask.sum(): {fg_mask.sum().item()}")


        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        print("[DEBUG] iou min/max:", iou.min().item(), iou.max().item())

        
        # 处理IoU损失可能的NaN
        if torch.isnan(loss_iou).any() or torch.isinf(loss_iou).any():
            LOGGER.warning(f"SafeBboxLoss: IoU损失为{loss_iou.item()}，设置为0!")
            loss_iou = torch.tensor(0.0).to(pred_bboxes_pos.device)
        
        # 如果不使用DFL，直接返回IoU损失
        loss_dfl = torch.tensor(0.0).to(pred_dist.device)
        if not self.use_dfl:
            return loss_iou, loss_dfl
        
        # 计算DFL损失
        try:
            # target_ltrb: shape [num_fg, 4]
            target_ltrb = bbox2dist(anchor_points_pos, target_bboxes_pos, self.reg_max)
            target_ltrb = target_ltrb.view(-1)
            
            # reshape pred_dist → [num_fg * 4, reg_max + 1]
            pred_dfl = pred_dist_pos.view(-1, 4, self.reg_max).reshape(-1, self.reg_max)  

            
            # flatten target → [num_fg * 4]
            target_dfl = target_ltrb.view(-1)
            print(f"[DEBUG] pred_dfl.shape = {pred_dfl.shape}")       # [N×4, 16]
            print(f"[DEBUG] target_ltrb.shape = {target_ltrb.shape}") # [N×4]
            print(f"[DEBUG] target_ltrb range: min={target_ltrb.min().item()}, max={target_ltrb.max().item()}")


            # 计算 DFL
            loss_dfl = self._df_loss(pred_dfl, target_dfl)

            # 应用权重（广播）
            weight = weight.expand(-1, 4).reshape(-1)
            loss_dfl = (loss_dfl * weight).sum() / target_scores_sum

            if torch.isnan(loss_dfl) or torch.isinf(loss_dfl):
                LOGGER.warning(f"SafeBboxLoss: DFL损失为{loss_dfl.item()}，设置为0!")
                loss_dfl = torch.tensor(0.0).to(pred_dist.device)
        except Exception as e:
            LOGGER.error(f"SafeBboxLoss: 计算DFL损失时出错: {e}")
            import traceback
            LOGGER.error(traceback.format_exc())
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)
        
        return loss_iou, loss_dfl
        
    @staticmethod
    def _df_loss(pred_dfl, target_ltrb):
        """分布式回归的 DFL 损失（支持 NaN 检查）"""
        if pred_dfl.numel() == 0 or target_ltrb.numel() == 0:
            return torch.tensor(0.0, device=pred_dfl.device)

        if torch.isnan(pred_dfl).any() or torch.isinf(pred_dfl).any() or torch.isnan(target_ltrb).any() or torch.isinf(target_ltrb).any():
            pred_dfl = torch.nan_to_num(pred_dfl)
            target_ltrb = torch.nan_to_num(target_ltrb)

        try:
            target = torch.clamp(target_ltrb, 0, pred_dfl.shape[-1] - 1)
            target_left = target.long()
            target_right = torch.clamp(target_left + 1, max=pred_dfl.shape[-1] - 1)

            weight_left = target_right.float() - target
            weight_right = target - target_left.float()

            loss_left = F.cross_entropy(pred_dfl, target_left, reduction="none") * weight_left
            loss_right = F.cross_entropy(pred_dfl, target_right, reduction="none") * weight_right

            return loss_left + loss_right
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"SafeBboxLoss._df_loss错误: {e}")
            return torch.tensor(0.0, device=pred_dfl.device)




class TomatoDetectWithRankLoss(v8DetectionLoss):
    """
    番茄检测与排序损失计算类
    支持cluster_ids和h_rel，增加了排序损失
    """

    # 特殊值定义
    UNKNOWN_H = -1.0  # 未知高度
    BUNCH_H = -2.0  # 整串番茄的特殊高度标记
    FRUIT_CLASSES = 4  # 单果类别数量，超过该值表示整串类型

    def __init__(self, model, lambda_rank=0.2, margin=0.1, tal_topk=10):
        """
        初始化番茄检测与排序损失计算类
        
        Args:
            model: 模型对象
            lambda_rank: 排序损失的权重系数
            margin: 排序损失的间隔阈值
            tal_topk: TaskAlignedAssigner的topk参数
        """
        super().__init__(model, tal_topk=tal_topk)
        from ultralytics.nn.modules.block import DFL  # 确保正确引入
        self.DFL = DFL(self.reg_max + 1)  # ⚠️ 是 DFL 不是 DFLoss
        self.lambda_rank = lambda_rank  # 排序损失的权重系数
        self.margin = margin  # 排序损失的间隔阈值
        
        # 创建SafeBboxLoss实例
        reg_max = 16
        if hasattr(model, 'model') and hasattr(model.model[-1], 'reg_max'):
            reg_max = model.model[-1].reg_max
        self.bbox_loss = SafeBboxLoss(reg_max).to(self.device)
        
        LOGGER.info(f"初始化番茄检测与排序损失，lambda_rank={lambda_rank}, margin={margin}, tal_topk={tal_topk}")
        LOGGER.info(f"使用安全版边界框损失计算器: SafeBboxLoss(reg_max={reg_max})")
        
        # 总是使用DFL
        self.use_dfl = True
        
    def bce(self, pred_scores, target_scores, weight=None, fg_mask=None):
        """
        带掩码的二元交叉熵损失
        
        Args:
            pred_scores: 预测分数 [batch_size, num_anchors, num_classes]
            target_scores: 目标分数 [batch_size, num_anchors, num_classes]
            weight: 可选权重 [batch_size, num_anchors, num_classes]
            fg_mask: 前景掩码 [batch_size, num_anchors]
            
        Returns:
            损失: 交叉熵损失
        """
        # 如果没有前景掩码，使用所有点
        if fg_mask is None:
            # 仅在目标分数不为零的位置计算BCE损失
            mask = target_scores.sum(dim=-1) > 0
            if mask.sum() == 0:
                return torch.tensor(0.0, device=pred_scores.device, requires_grad=True)
                
            # 计算损失
            loss = F.binary_cross_entropy_with_logits(
                pred_scores[mask], target_scores[mask], reduction='none'
            )
            
            # 应用权重
            if weight is not None:
                loss = loss * weight[mask]
                
            return loss
        else:
            # 使用前景掩码
            # 首先确保掩码维度匹配
            if fg_mask.dim() == 2:
                fg_mask = fg_mask.unsqueeze(-1).expand_as(target_scores)
            
            # 仅在前景位置计算BCE损失
            if fg_mask.sum() == 0:
                return torch.tensor(0.0, device=pred_scores.device, requires_grad=True)
                
            # 计算损失
            loss = F.binary_cross_entropy_with_logits(
                pred_scores[fg_mask], target_scores[fg_mask], reduction='none'
            )
            
            # 应用权重
            if weight is not None:
                loss = loss * weight[fg_mask]
                
            return loss
    
    def decode_dfl(self, pred_distri):
        """
        将预测的分布映射为连续的 ltrb 距离。

        Args:
            pred_distri: Tensor of shape [B, N, 4 * reg_max]

        Returns:
            Tensor: Decoded distances of shape [B, N, 4]
        """
        B, N, _ = pred_distri.shape
        pred = pred_distri.view(B, N, 4, self.reg_max)  # [B, N, 4, reg_max]
        pred_prob = F.softmax(pred, dim=-1)
        proj = torch.arange(self.reg_max, dtype=pred.dtype, device=pred.device)
        pred_ltrb = (pred_prob * proj).sum(-1)  # [B, N, 4]
        return pred_ltrb


    def __call__(self, preds, batch):
        """
        Forward pass through the loss function.
        
        Args:
            preds (torch.Tensor | dict): Predictions from the model.
            batch (dict): Batch data containing images and labels.
            
        Returns:
            tuple: A tuple containing the total loss and individual loss items.
        """
        # =================== CHECKPOINT A ===================
        print("\n" + "="*50)
        print("🎯 CHECKPOINT A: Raw Input Batch Data")
        if "bboxes" in batch and batch["bboxes"].numel() > 0:
            print(f"    batch['bboxes'] shape: {batch['bboxes'].shape}")
            print(f"    batch['bboxes'] sample:\n{batch['bboxes'][:5]}")
            print(f"    Value range: [{batch['bboxes'].min().item():.4f}, {batch['bboxes'].max().item():.4f}]")
        else:
            print("    batch['bboxes'] is empty or not found!")
        print("="*50 + "\n")
        # ====================================================
        # 规范化预测和提取核心特征
        feats, pred_h_pos = self._normalize_predictions(preds)
        if feats is None:
            # 预测结果无效，返回零损失
            zero_loss = torch.zeros(1, device=self.device)
            return zero_loss, torch.zeros(4, device=self.device)
            
        # 从特征中提取预测分布和分数
        pred_distri, pred_scores = self._extract_predictions(feats)
        
        # 获取批次大小和图像大小
        batch_size = pred_scores.shape[0]
        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        
        # 创建锚点网格和步长张量
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        
        # 准备目标数据
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size)
        # =================== CHECKPOINT B ===================
        scale_tensor_check = imgsz[[1, 0, 1, 0]]
        print("\n" + "="*50)
        print("🎯 CHECKPOINT B: Preprocess Scaling Factor")
        print(f"    Shape of first feature map (feats[0].shape): {feats[0].shape}")
        print(f"    Model stride (self.stride[0]): {self.stride[0]}")
        print(f"    Calculated image size (imgsz): {imgsz}")
        print(f"    Resulting scale_tensor: {scale_tensor_check}")
        print("="*50 + "\n")
        # ====================================================

        
        # 分离标签和边界框  
        
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)  # 非零边界框的掩码
        # =================== CHECKPOINT 1 ===================
        if gt_bboxes.numel() > 0:
            print("\n" + "="*50)
            print("✅ CHECKPOINT 1: Ground-Truth Bboxes (`gt_bboxes`)")
            print(f"    gt_bboxes shape: {gt_bboxes.shape}")
            print(f"    Value range: [{gt_bboxes.min().item():.2f}, {gt_bboxes.max().item():.2f}]")
            print("="*50 + "\n")
        # ====================================================
        
        # 将预测边界框形式转换为XYXY格式
        # 正确方式：先解码DFL，再传入dist2bbox
        pred_ltrb = self.decode_dfl(pred_distri)  # [B, N, 4]
        pred_bboxes = dist2bbox(pred_ltrb, anchor_points * stride_tensor.unsqueeze(0), xywh=False)  # [B, N, 4]


        print("🧪 pred_bboxes: ", pred_bboxes.shape)  # [bs, num_anchors, 4]
        print("pred_bboxes sample:", pred_bboxes[0, :5])  # 查看前5个预测框
        print("    x range: [{:.4f}, {:.4f}]".format(pred_bboxes.min().item(), pred_bboxes.max().item()))
        

        # 分配正样本
        
        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
                pred_scores.detach().sigmoid(),
                pred_bboxes.detach(),
                anchor_points * stride_tensor,
                gt_labels,
                gt_bboxes,
                mask_gt
            )
        print(f"🟡 正样本数量: {fg_mask.sum()} / {fg_mask.numel()}")
        print(f"🟡 target_scores sum: {target_scores.sum()}")
        print(f"🟡 target_bboxes shape: {target_bboxes.shape}")
        print(f"🟡 target_bboxes sample: {target_bboxes[0, :5]}")



        # 计算损失
        if target_bboxes.shape[-1] == 4 and stride_tensor is not None:
                # 🎯 使用切片方式的修正
                # stride_tensor 形状: [N, 2]
                # 添加一个批次维度，使其变为 [1, N, 2]，以便与 [B, N, 2] 的切片进行广播
                stride_tensor_exp = stride_tensor.unsqueeze(0)
                
                # 分别处理 (x1, y1) 和 (x2, y2)
                target_bboxes[..., 0:2] /= stride_tensor_exp
                target_bboxes[..., 2:4] /= stride_tensor_exp
            # 计算边界框损失
        print("🎯 aligned target_bboxes sample:", target_bboxes[0, :5])
        print("🎯 pred_bboxes sample:", pred_bboxes[0, :5])

        loss_bbox, loss_dfl = self.bbox_loss(
            pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_gt_idx, fg_mask, stride_tensor
            )

        # 计算分类损失
        try:
            if fg_mask.sum():
                # 有前景点，正常计算分类损失
                weight = torch.ones_like(target_scores)
                loss_cls = F.binary_cross_entropy_with_logits(
                    pred_scores[fg_mask],
                    target_scores[fg_mask],
                    weight=weight[fg_mask],
                    reduction='none'
                ).sum() / max(fg_mask.sum(), 1)  # ✅

            else:
                # 没有前景点，使用一个小的分类损失
                loss_cls = torch.tensor(1.0, device=self.device, requires_grad=True)
        except Exception as e:
            LOGGER.error(f"计算分类损失时出错: {e}")
            import traceback
            LOGGER.error(traceback.format_exc())
            loss_cls = torch.tensor(1.0, device=self.device, requires_grad=True)
        
        # 计算排序损失
        try:
            loss_rank = self.compute_ranking_loss(batch, pred_h_pos) * self.lambda_rank
        except Exception as e:
            LOGGER.error(f"计算排序损失时出错: {e}")
            import traceback
            LOGGER.error(traceback.format_exc())
            loss_rank = torch.tensor(0.01, device=self.device, requires_grad=True)
        
        # 确保所有损失都有梯度
        if not loss_bbox.requires_grad:
            loss_bbox = torch.tensor(0.0, device=self.device, requires_grad=True)
        if not loss_cls.requires_grad:
            loss_cls = torch.tensor(1.0, device=self.device, requires_grad=True)
        if not loss_dfl.requires_grad:
            loss_dfl = torch.tensor(0.0, device=self.device, requires_grad=True)
        if not loss_rank.requires_grad:
            loss_rank = torch.tensor(0.01, device=self.device, requires_grad=True)
        
        # 计算总损失
        loss = loss_bbox + loss_cls + loss_dfl + loss_rank
        

        
        # 返回总损失和各损失分量
        return loss, torch.stack((loss_bbox, loss_cls, loss_dfl, loss_rank))

    def _normalize_predictions(self, preds):
        """
        规范化预测输入格式
        
        Args:
            preds: 预测结果，可以是以下格式之一：
                - 张量列表: [feature1, feature2, ...]
                - 元组: (feats, pred_h_pos) 或 (NULL, feats, pred_h_pos)
                - 字典: {"features": feats, "h_pos": pred_h_pos}
                - Detect_Efficient_Tomato的特殊输出: (y, {"features": feats, "h_pos": h_pos})
                
        Returns:
            feats: 特征列表
            pred_h_pos: 高度预测，如果不存在则为None
        """
        feats = None
        pred_h_pos = None
        
        LOGGER.info(f"规范化预测输入，类型: {type(preds)}")
        
        # 处理字典格式（训练模式下的Detect_Efficient_Tomato输出）
        if isinstance(preds, dict):
            LOGGER.info(f"处理字典格式输入，包含键: {list(preds.keys())}")
            if "features" in preds and isinstance(preds["features"], list):
                feats = preds["features"]
                pred_h_pos = preds.get("h_pos", None)
                LOGGER.info(f"从字典中提取: features={type(feats)}, 特征数量={len(feats) if feats is not None else 0}")
                LOGGER.info(f"从字典中提取: h_pos={type(pred_h_pos)}, h_pos特征数量={len(pred_h_pos) if isinstance(pred_h_pos, list) else 0}")
        
        # 处理元组/列表格式
        elif isinstance(preds, (list, tuple)):
            LOGGER.info(f"处理元组/列表格式输入，长度: {len(preds)}")
            
            # 推理模式下的Detect_Efficient_Tomato输出
            if len(preds) == 2 and isinstance(preds[1], dict) and "features" in preds[1]:
                LOGGER.info("检测到Detect_Efficient_Tomato的特殊输出格式: (y, {'features': feats, 'h_pos': h_pos})")
                feats = preds[1].get("features")
                pred_h_pos = preds[1].get("h_pos")
                LOGGER.info(f"从特殊输出中提取: features={type(feats)}, 特征数量={len(feats) if isinstance(feats, list) else 0}")
                LOGGER.info(f"从特殊输出中提取: h_pos={type(pred_h_pos)}, h_pos特征数量={len(pred_h_pos) if isinstance(pred_h_pos, list) else 0}")
            
            # 标准YOLO格式: [feature1, feature2, ...]
            elif len(preds) > 0 and all(isinstance(p, torch.Tensor) for p in preds):
                LOGGER.info("检测到标准YOLO特征列表格式")
                feats = preds
                
            # 嵌套列表结构: [[tensor1, tensor2, ...], ...]
            elif len(preds) > 0 and isinstance(preds[0], list) and len(preds[0]) > 0 and all(isinstance(x, torch.Tensor) for x in preds[0]):
                LOGGER.info("检测到嵌套列表格式: [[tensor1, tensor2, ...], ...]")
                feats = preds[0]
                if len(preds) >= 2 and isinstance(preds[1], list) and all(isinstance(x, torch.Tensor) for x in preds[1]):
                    pred_h_pos = preds[1]
            
            # 前导None的格式: (None, [tensor1, tensor2, ...], ...)
            elif len(preds) > 1 and preds[0] is None and isinstance(preds[1], list):
                LOGGER.info("检测到前导None的格式: (None, [tensor1, tensor2, ...], ...)")
                feats = preds[1]
                if len(preds) > 2 and isinstance(preds[2], list):
                    pred_h_pos = preds[2]
        
        # 处理单个张量
        elif isinstance(preds, torch.Tensor):
            LOGGER.info("处理单个张量输入")
            feats = [preds]
        
        # 验证特征是否有效
        if feats is None:
            LOGGER.warning(f"无法识别的预测输入格式: {type(preds)}")
            return None, None
            
        if not isinstance(feats, list):
            LOGGER.warning(f"特征不是列表类型: {type(feats)}")
            return None, None
            
        if len(feats) == 0:
            LOGGER.warning("空特征列表")
            return None, None
            
        if not all(isinstance(f, torch.Tensor) for f in feats):
            LOGGER.warning(f"特征列表包含非张量元素: {[type(f) for f in feats]}")
            return None, None
        
        # 验证并打印特征信息
        LOGGER.info(f"规范化完成，特征数量: {len(feats)}")
        for i, feat in enumerate(feats):
            LOGGER.info(f"特征[{i}]: 形状={feat.shape}, 类型={feat.dtype}")
        
        # 验证并打印h_pos信息
        if pred_h_pos is not None and isinstance(pred_h_pos, list) and len(pred_h_pos) > 0:
            LOGGER.info(f"h_pos特征数量: {len(pred_h_pos)}")
            for i, h_pos in enumerate(pred_h_pos):
                if isinstance(h_pos, torch.Tensor):
                    LOGGER.info(f"h_pos[{i}]: 形状={h_pos.shape}, 类型={h_pos.dtype}")
                else:
                    LOGGER.warning(f"h_pos[{i}]不是张量: {type(h_pos)}")
        
        return feats, pred_h_pos
    
    def _extract_predictions(self, feats):
        """
        从特征中提取预测分布和分数
        
        Args:
            feats: 特征列表
            
        Returns:
            pred_distri: 预测分布
            pred_scores: 预测分数
        """
        try:
            # 特征拼接前获取批次大小和通道数
            batch_size = feats[0].shape[0]
            total_channels = feats[0].shape[1]
            
            # 打印通道数信息
            LOGGER.info(f"特征通道数: {total_channels}, self.no: {self.no}")
            
            # 检查通道数是否匹配
            if total_channels > self.no:
                LOGGER.warning(f"通道数不匹配: 特征通道数={total_channels}, self.no={self.no}, 可能缺少高度预测通道")
                LOGGER.warning(f"假设特征形式为: 4*reg_max(box) + nc(class) + 额外高度通道, 检查通道数是否为 {4*self.reg_max + self.nc + 1}")
                
                # 计算预期通道数
                expected_box_channels = 4 * self.reg_max  # 边界框通道
                expected_cls_channels = self.nc  # 类别通道
                expected_h_pos_channels = 1  # 高度通道
                expected_total = expected_box_channels + expected_cls_channels + expected_h_pos_channels
                
                if total_channels == expected_total:
                    LOGGER.info(f"通道数符合预期: {expected_total} = {expected_box_channels}(box) + {expected_cls_channels}(class) + {expected_h_pos_channels}(h_pos)")
                    
                    # 拼接特征并正确分割
                    features_cat = torch.cat([xi.view(batch_size, total_channels, -1) for xi in feats], 2)
                    pred_distri = features_cat[:, :expected_box_channels]
                    pred_scores = features_cat[:, expected_box_channels:expected_box_channels+expected_cls_channels]
                    
                    # 调整维度顺序
                    pred_scores = pred_scores.permute(0, 2, 1).contiguous()
                    pred_distri = pred_distri.permute(0, 2, 1).contiguous()
                    
                    LOGGER.info(f"调整后: pred_distri: {pred_distri.shape}, pred_scores: {pred_scores.shape}")
                    return pred_distri, pred_scores
            
            # 原始方法 - 只考虑 self.no 通道
            LOGGER.info(f"使用默认方法提取预测，特征视图使用 self.no={self.no} 通道")
            pred_distri, pred_scores = torch.cat([xi.view(batch_size, self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)
            
            LOGGER.info(f"pred_distri: {pred_distri.shape}, pred_scores: {pred_scores.shape}")
            # 调整维度顺序
            pred_scores = pred_scores.permute(0, 2, 1).contiguous()
            pred_distri = pred_distri.permute(0, 2, 1).contiguous()
            LOGGER.info(f"调整后: pred_distri: {pred_distri.shape}, pred_scores: {pred_scores.shape}")
            return pred_distri, pred_scores
        except Exception as e:
            LOGGER.error(f"提取预测时出错: {e}")
            import traceback
            LOGGER.error(traceback.format_exc())
            raise
    


            
    def compute_ranking_loss(self, batch, pred_scores=None):
        """
        计算番茄串内成熟度排序损失
        
        基于观察：同一串内，位置更高的番茄成熟度更高
        
        Args:
            batch: 数据批次，包含：
                - batch["cluster_ids"]: 串ID标签
                - batch["h_rel"]: 相对高度标签
                - batch["cls"]: 类别标签
            pred_scores: 预测的类别得分，用于可能的未来扩展
        
        Returns:
            rank_loss: 排序损失值
        """
        # 设置一个最小排序损失，确保即使所有样本对都符合规则，也返回一个小的正值
        MIN_RANK_LOSS = 0.01
        
        # 检查是否有必要的数据
        if "cluster_ids" not in batch or "h_rel" not in batch or "cls" not in batch:
            # LOGGER.warning("排序损失: 批次中缺少必要的字段")
            return torch.tensor(MIN_RANK_LOSS, device=self.device)
            
        if batch["cluster_ids"] is None or batch["h_rel"] is None or batch["cls"] is None:
            # LOGGER.warning("排序损失: 批次中有字段为None")
            return torch.tensor(MIN_RANK_LOSS, device=self.device)
            
        if batch["cluster_ids"].numel() == 0 or batch["h_rel"].numel() == 0 or batch["cls"].numel() == 0:
            # LOGGER.warning("排序损失: 批次中有空字段")
            return torch.tensor(MIN_RANK_LOSS, device=self.device)
            
        # 准备数据
        cluster_ids = batch["cluster_ids"].flatten()  # 展平为一维
        h_rel = batch["h_rel"].flatten()  # 展平为一维
        classes = batch["cls"]  # 类别
        
        # 检查并过滤无效数据
        valid_mask = (cluster_ids >= 0) & (h_rel >= 0) & (h_rel <= 1)
        if valid_mask.sum() == 0:
            # LOGGER.warning("排序损失: 没有有效的数据")
            return torch.tensor(MIN_RANK_LOSS, device=self.device)
            
        # 应用过滤
        valid_cluster_ids = cluster_ids[valid_mask]
        valid_h_rel = h_rel[valid_mask]
        valid_classes = classes[valid_mask]
            
        # 获取所有唯一的串ID
        unique_clusters = valid_cluster_ids.unique()
        
        # 初始化损失
        rank_loss = torch.tensor(0.0, device=self.device)
        pair_count = 0
        penalty_count = 0
        
        # 对每个串内的番茄进行排序比较
        for cluster_id in unique_clusters:
            # 跳过ID小于0的串（通常是背景或无效串）
            if cluster_id < 0:
                    continue
                    
            # 找到当前串的所有番茄
            cluster_mask = valid_cluster_ids == cluster_id
            cluster_h_rel = valid_h_rel[cluster_mask]
            cluster_classes = valid_classes[cluster_mask]
            
            n_tomatoes = len(cluster_h_rel)
            if n_tomatoes <= 1:
                continue  # 跳过只有一个番茄的串
                
            # 比较同一串内的所有番茄对
            for i in range(n_tomatoes):
                for j in range(i + 1, n_tomatoes):
                    h1, h2 = cluster_h_rel[i], cluster_h_rel[j]
                    cls1, cls2 = cluster_classes[i], cluster_classes[j]
                    
                    # 计算高度差异
                    h_diff = torch.abs(h1 - h2)
                    
                    # 如果高度差异显著（大于阈值）
                    if h_diff > self.margin:
                            pair_count += 1
            
                        # 确定预期的排序关系
                    higher_tomato, lower_tomato = (i, j) if h1 < h2 else (j, i)
                    higher_cls = cluster_classes[higher_tomato]
                    lower_cls = cluster_classes[lower_tomato]
                        
                        # 如果类别差异与高度差异不符，增加损失
                        # 假设数字越小表示越成熟（在同一类别组内）
                    if higher_cls < lower_cls:
                            # 成熟度排序与高度排序一致，符合预期
                            pass
                    else:
                            # 成熟度排序与高度排序不一致，应该惩罚
                            # 计算间隔损失：max(0, margin - |cls1 - cls2|)
                            cls_diff = torch.abs(higher_cls - lower_cls).float()
                            loss = torch.nn.functional.relu(self.margin - h_diff)
                            rank_loss += loss
                            penalty_count += 1
        
        # 如果有采样对，计算平均损失；否则使用最小损失
            if pair_count > 0:
                rank_loss = rank_loss / pair_count
                # 确保损失不为零
                rank_loss = torch.max(rank_loss, torch.tensor(MIN_RANK_LOSS, device=self.device))
                LOGGER.info(f"排序损失: 总共检查了 {pair_count} 对样本，有 {pair_count - penalty_count} 对产生了常规损失，有 {penalty_count} 对产生了额外惩罚")
            else:
                # LOGGER.info("排序损失: 没有有效的样本对，返回最小排序损失")
                rank_loss = torch.tensor(MIN_RANK_LOSS, device=self.device)
            
        return rank_loss
