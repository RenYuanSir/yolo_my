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

    def preprocess(self, targets, batch_size, scale_tensor):
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
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
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

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

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
        
    def __call__(self, pred_dist, pred_bboxes, anchor_points, stride_tensor, target_bboxes, target_scores, target_scores_sum, fg_mask, eps=1e-8):
        """
        计算边界框损失和DFL损失
        
        Args:
            pred_dist (tensor): 预测的分布
            pred_bboxes (tensor): 预测的边界框
            anchor_points (tensor): 锚点
            stride_tensor (tensor): 步长张量
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
              
        # 确保fg_mask是布尔类型
        fg_mask = fg_mask.bool()
        
        # 检查维度匹配问题
        if pred_bboxes.dim() == 3 and fg_mask.dim() == 2:
            B = pred_bboxes.shape[0]  # 获取批次大小
            # 将fg_mask扩展到匹配pred_bboxes的批次维度
            expanded_fg_mask = fg_mask.unsqueeze(0).expand(B, -1, -1)
            # 使用view调整形状以匹配索引需求
            expanded_fg_mask = expanded_fg_mask.view(B, -1)
            
            # 使用expanded_fg_mask处理批次维度的索引
            pred_bboxes_list = []
            pred_dist_list = []
            for i in range(B):
                batch_mask = expanded_fg_mask[i]
                if batch_mask.sum() > 0:
                    pred_bboxes_list.append(pred_bboxes[i, batch_mask])
                    pred_dist_list.append(pred_dist[i, batch_mask])
            
            # 如果存在有效预测，则合并结果
            if pred_bboxes_list:
                pred_bboxes_pos = torch.cat(pred_bboxes_list, dim=0)
                pred_dist_pos = torch.cat(pred_dist_list, dim=0)
            else:
                # 如果没有有效预测，创建空张量
                return torch.tensor(0.0).to(pred_dist.device), torch.tensor(0.0).to(pred_dist.device)
        else:
            # 原始索引方式，适用于维度匹配的情况
            pred_bboxes_pos = pred_bboxes[fg_mask]
            pred_dist_pos = pred_dist[fg_mask]
          
        target_bboxes_pos = target_bboxes[fg_mask] 
        # scale target_bboxes from [0,1] → [0, image_size]
        print("\n" + "="*60)
        print("🕵️  DEBUGGING INSIDE SafeBboxLoss")
        print(f"    Number of positive pairs to compare: {pred_bboxes_pos.shape[0]}")
        if pred_bboxes_pos.shape[0] > 0:
            # Print the first pair for direct comparison
            print(f"    Sample PRED_BBOX_POS:   {pred_bboxes_pos[0].tolist()}")
            print(f"    Sample TARGET_BBOX_POS: {target_bboxes_pos[0].tolist()}")
            
            # Check for invalid boxes where x1 >= x2 or y1 >= y2
            invalid_preds = (pred_bboxes_pos[:, 0] >= pred_bboxes_pos[:, 2]) | (pred_bboxes_pos[:, 1] >= pred_bboxes_pos[:, 3])
            if invalid_preds.any():
                print(f"    ❌ WARNING: Found {invalid_preds.sum()} invalid predicted boxes (x1>=x2 or y1>=y2)!")
        print("="*60 + "\n")
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
            # 1. 计算原始的、巨大的像素距离
            target_ltrb_pixel = bbox2dist(anchor_points_pos, target_bboxes_pos, self.reg_max - 1)

            # 2. 获取正样本对应的步长(stride)
            #    stride_tensor 形状为 [1, N_anchors], 我们需要扩展并筛选
            strides_pos = stride_tensor.t().expand(B, -1)[fg_mask] # 形状: [num_fg]

            # 3. !! 进行尺度翻译：将像素距离除以步长 !!
            #    unsqueeze(-1) 是为了广播 [num_fg, 1] 到 [num_fg, 4]
            target_ltrb_scaled = target_ltrb_pixel / strides_pos.unsqueeze(-1)
            
            # 4. 用缩放后的、DFL能理解的目标来计算损失
            #    reshape pred_dist → [num_fg * 4, reg_max]
            pred_dfl = pred_dist_pos.view(-1, self.reg_max)
            target_dfl = target_ltrb_scaled.view(-1)
            
            loss_dfl = self._df_loss(pred_dfl, target_dfl)
            
            # 应用权重（广播）
            # weight 是 [num_fg, 1], 扩展并重塑为 [num_fg * 4]
            weight_dfl = weight.expand(-1, 4).reshape(-1)
            loss_dfl = (loss_dfl * weight_dfl).sum() / target_scores_sum

            if torch.isnan(loss_dfl) or torch.isinf(loss_dfl):
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

    def __init__(self, model, lambda_rank=0.2, margin=0.05, tal_topk=10):
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
        self.focal_loss = FocalLoss()
        self.use_dfl = self.reg_max > 1
        self.proj = torch.arange(self.reg_max, dtype=torch.float).to(self.device)
        self.bbox_decode = super().bbox_decode
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        
        LOGGER.info(f"初始化番茄检测与排序损失，lambda_rank={lambda_rank}, margin={margin}, tal_topk={tal_topk}")
        LOGGER.info(f"使用标准v8DetectionLoss中的边界框损失计算器: BboxLoss(reg_max={self.reg_max})")
        

    


    def __call__(self, preds, batch):
        """
        该方法整合了YOLOv8官方的v8DetectionLoss.__call__的标准流程，
        并精确地注入了自定义的rank_loss计算，确保了所有基础计算的正确性。
        """
        print("🕵️  DEBUGGING INSIDE TomatoDetectWithRankLoss")
        print(f"    raw input bbox sample 5: {batch['bboxes'][:5]}")
        feats = preds["features"]
        pred_h_pos = preds["h_pos"]

        # 2. 以下完全遵循官方代码，进行预测的解包和准备工作
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # 3. 准备真值 (Targets)，并将其转换为像素级坐标
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        # 关键：创建并传递 scale_tensor，将真值框反归一化为像素级
        scale_tensor = torch.tensor([imgsz[1], imgsz[0], imgsz[1], imgsz[0]], device=self.device)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor)
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy (现在是像素级)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # 4. 解码预测框 (官方流程)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # 输出归一化的 xyxy

        # 5. 正样本分配 (官方流程) - 在此环节将预测框和锚点放大到像素级
        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype), # 放大到像素级
            anchor_points * stride_tensor, # 放大到像素级
            gt_labels,
            gt_bboxes, # 本身已经是像素级
            mask_gt,
        )

        print("🕵️  DEBUGGING INSIDE TomatoDetectWithRankLoss")
        print(f"    pred_bboxes sample 5: {pred_bboxes[:5]}")
        # 确保fg_mask是布尔类型
        fg_mask = fg_mask.bool()
        
        # 计算目标分数总和用于归一化
        target_scores_sum = max(target_scores.sum(), 1)

        # 6. 计算标准损失
        if fg_mask.sum():
            # 使用父类的bbox_loss计算器
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask)
                
            # 计算分类损失（模仿父类v8DetectionLoss中的实现方式）
            cls_loss = self.bce(pred_scores, target_scores.to(dtype))  # BCE(pred_scores, target_scores)
            loss[1] = cls_loss.sum() / target_scores_sum
        
        # 7. 【注入自定义的 Rank Loss】
        #    这部分逻辑现在可以安全地执行，因为 fg_mask 和 target_gt_idx 是正确的
        pred_h_cat = torch.cat([p.view(p.shape[0], 1, -1) for p in pred_h_pos], dim=2)
        pred_h_tensor = pred_h_cat.permute(0, 2, 1).contiguous()
        
        # 准备对齐的自定义真值 (这部分逻辑不变，但现在它服务于一个能正常工作的流程)
        custom_data = torch.cat(
            (batch['cls'].view(-1, 1).to(self.device), batch['cluster_ids'].view(-1, 1).to(self.device), batch['h_rel'].view(-1, 1).to(self.device)), 1
        ).to(self.device)
        custom_targets = torch.zeros(batch_size, targets.shape[1], 3, device=self.device)
        for i in range(batch_size):
            mask = (batch['batch_idx'] == i)
            count = mask.sum()
            if count > 0:
                custom_targets[i, :count] = custom_data[mask]
        
        # 计算 Rank Loss (lambda_rank > 0 时生效)
        loss_rank = self.compute_ranking_loss(pred_h_tensor, custom_targets, fg_mask, target_gt_idx)
        
        # 8. 计算总损失
        loss_total = (loss[0] + loss[1] + loss[2]) + loss_rank * self.lambda_rank

        return loss_total, torch.cat((loss, loss_rank.unsqueeze(0))).detach()

    def compute_ranking_loss(self, pred_h, custom_targets, fg_mask, target_gt_idx):
        """
        Computes ranking loss using a vectorized approach to prevent hanging.
        """
        batch_size = pred_h.shape[0]
        device = pred_h.device
        
        # 确保fg_mask是布尔类型
        fg_mask = fg_mask.bool()
        
        if fg_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        # Get aligned predictions and ground truths for positive anchors
        positive_pred_h = pred_h[fg_mask] # Shape: [num_pos]
        assigned_gt_indices = target_gt_idx[fg_mask]
        batch_idx_grid = torch.arange(batch_size, device=device).view(batch_size, 1).repeat(1, pred_h.shape[1])
        positive_batch_idx = batch_idx_grid[fg_mask] # 得到 [num_pos]
        aligned_gts = custom_targets[positive_batch_idx, assigned_gt_indices]
        gt_classes = aligned_gts[:, 0]
        gt_cluster_ids = aligned_gts[:, 1]
        gt_h_rel = aligned_gts[:, 2]

        unique_clusters = torch.unique(gt_cluster_ids)
        final_loss = torch.tensor(0.0, device=self.device)
        total_pairs = 0

        for cluster_id in unique_clusters:
            if cluster_id < 0:
                continue
            
            # Find all items belonging to the current cluster
            cluster_mask = gt_cluster_ids == cluster_id
            n_tomatoes = cluster_mask.sum()
            if n_tomatoes < 2:
                continue
                
            # Get the data for the current cluster
            cluster_pred_h = positive_pred_h[cluster_mask]
            cluster_gt_h = gt_h_rel[cluster_mask]
            cluster_classes = gt_classes[cluster_mask]
            
            # --- VECTORIZED PAIR GENERATION ---
            # Create all unique pairs of indices, e.g., (0,1), (0,2), (1,2)...
            indices = torch.arange(n_tomatoes, device=self.device)
            p1_idx, p2_idx = torch.combinations(indices, r=2).unbind(1)
            
            # Gather data for all pairs at once
            h_pred1, h_pred2 = cluster_pred_h[p1_idx].squeeze(-1), cluster_pred_h[p2_idx].squeeze(-1)
            h_gt1, h_gt2 = cluster_gt_h[p1_idx], cluster_gt_h[p2_idx]
            cls1, cls2 = cluster_classes[p1_idx], cluster_classes[p2_idx]

            # --- VECTORIZED LOSS CALCULATION ---
            # Create a target tensor: 1 if h_gt1 < h_gt2, -1 if h_gt2 < h_gt1, 0 otherwise
            target = torch.sign(h_gt2 - h_gt1)
            base_margin = 0.2
            dynamic_margin = base_margin + self.margin * torch.abs(h_gt1 - h_gt2)
            # Calculate base ranking loss for all pairs at once
            # margin_ranking_loss wants to make pred1 > pred2 for target=1
            loss = torch.clamp_min(-target * (h_pred2 - h_pred1) + dynamic_margin, 0)
            regression_loss = F.smooth_l1_loss(cluster_pred_h.squeeze(-1), cluster_gt_h, reduction='mean') * 5.0
            
            # --- 可导的重惩罚项 ---

            # 定义一个超参数 k 来控制 sigmoid 的陡峭程度。k越大，函数越接近一个硬性开关。
            # 这是一个需要根据实验效果调整的超参数。
            k = 10.0 

            # 基础惩罚：给所有样本的损失都乘以一个基础权重 (例如 2.0)
            # 原来的 penalty_factor 是 2.0 和 5.0，所以额外惩罚是 3.0
            base_penalty_factor = 2.0
            heavy_penalty_additive = 3.0 # 5.0 - 2.0

            penalized_loss = loss * base_penalty_factor

            # 条件1: GT 要求 h1 < h2 (target=1)，但模型错误地预测 h1 >= h2
            # 我们只在需要重惩罚的类别组合上应用 (cls1=0, cls2=1)
            heavy_penalty_mask1 = (target == 1) & (cls1 == 0) & (cls2 == 1)

            # 计算"错误程度"分数：h_pred1 相对于 h_pred2 大了多少
            # sigmoid(k * (h_pred1 - h_pred2)) 会在 h_pred1 > h_pred2 时趋近于1
            wrongness_score1 = torch.sigmoid(k * (h_pred1 - h_pred2))

            # 计算可导的额外惩罚值
            # 只在满足条件 (heavy_penalty_mask1) 的样本上，施加与"错误程度"成正比的惩罚
            additional_penalty1 = heavy_penalty_additive * wrongness_score1 * heavy_penalty_mask1


            # 条件2: GT 要求 h2 < h1 (target=-1)，但模型错误地预测 h2 >= h1
            heavy_penalty_mask2 = (target == -1) & (cls2 == 0) & (cls1 == 1)
            wrongness_score2 = torch.sigmoid(k * (h_pred2 - h_pred1))
            additional_penalty2 = heavy_penalty_additive * wrongness_score2 * heavy_penalty_mask2


            # --- 更新 Final Loss ---
            # 将基础惩罚损失、可导的额外惩罚、回归损失相加
            # 注意，这里我们将惩罚项直接加到 loss 上，而不是作为系数
            final_loss += (penalized_loss + additional_penalty1 + additional_penalty2).sum() + regression_loss * n_tomatoes
            total_pairs += len(p1_idx)

        # Normalize the total loss
        if total_pairs > 0:
            return final_loss / total_pairs
            
        return torch.tensor(0.0, device=self.device)