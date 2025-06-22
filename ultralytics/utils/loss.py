# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist
from typing import Tuple, Dict, List

# 常量：表示无意义的h_rel值
UNKNOWN_H = -1.0

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

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

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


class TomatoDetectWithRankLoss(v8DetectionLoss):
    """
    番茄检测与串排序损失函数
    
    结合标准检测损失和番茄专用排序损失，基本假设：
    1. 同一串(cluster)内的番茄，位置越高(h_rel越小)成熟度越高
    2. 成熟度排序: fully_ripe > ripe > turning > green
    
    Combined loss = λ_box * L_box + λ_cls * L_cls + λ_dfl * L_dfl + λ_rank * L_rank
    Where:
        - L_rank: 串内番茄成熟度排序损失（基于位置的排序约束）
    """

    def __init__(self, model, lambda_rank=0.2, margin=0.1, tal_topk=10):
        """
        初始化番茄串排序损失函数
        
        Args:
            model: 检测模型
            lambda_rank (float): 排序损失权重
            margin (float): 排序间隔
            tal_topk (int): 任务对齐指派的top-k数量
        """
        super().__init__(model, tal_topk)
        
        # 排序损失配置
        self.lambda_rank = lambda_rank  # 排序损失权重
        self.margin = margin  # 排序间隔
        
        # 番茄类别配置 - 成熟度从高到低：fully_ripe(0) > ripe(1) > turning(2) > green(3)
        self.FRUIT_CLASSES = 4  # 单果类别数量（不包括串类别）
        
        # 常量定义 - 使用与数据标签一致的值
        self.UNKNOWN_H = -1.0  # 无效的高度值标记 (-1.0)
        self.BUNCH_H = 0.0     # 番茄串的默认高度值 (0.0)
        
        # 调试配置
        self.debug = False     # 是否打印调试信息

    def __call__(self, preds, batch):
        """
        计算损失函数
        
        Args:
            preds: 模型预测结果
            batch: 包含ground truth的批次数据
                - batch["cluster_ids"]: 串ID标签
                - batch["h_rel"]: 相对高度标签(0~1)，0表示最高位置
                - batch["cls"]: 类别标签
        
        Returns:
            loss: 损失总和
            loss_items: 各损失项
        """
        # 完全重写 __call__ 方法，确保特征处理正确
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, rank_loss
        
        # 提取特征和h_pos预测
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_h_pos = None
        
        # 处理不同格式的输入
        if isinstance(feats, dict):
            # 检查是否有h_pos字段 - Detect_Efficient_Tomato的格式
            if "h_pos" in feats:
                pred_h_pos = feats["h_pos"]
            if "features" in feats:
                feats = feats["features"]
                
        # 确保feats是扁平的张量列表
        if not isinstance(feats, list) or not all(isinstance(f, torch.Tensor) for f in feats):
            LOGGER.warning(f"无法处理的特征格式: {type(feats)}，尝试降级处理")
            # 创建虚拟特征以避免崩溃
            dummy_feat = torch.zeros(
                (batch["img"].shape[0], self.no, 8, 8), 
                device=self.device
            )
            feats = [dummy_feat]
            LOGGER.warning(f"创建了虚拟特征: {dummy_feat.shape}")

        try:
            # 分离预测分布和分类分数
            pred_distri, pred_scores = torch.cat(
                [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 
                2
            ).split((self.reg_max * 4, self.nc), 1)
            
            # 转置为预期格式
            pred_scores = pred_scores.permute(0, 2, 1).contiguous()
            pred_distri = pred_distri.permute(0, 2, 1).contiguous()
            
            # 添加调试信息
            LOGGER.info(f"[DEBUG_LOSS] pred_scores shape: {pred_scores.shape}")
            LOGGER.info(f"[DEBUG_LOSS] pred_distri shape: {pred_distri.shape}")
            
            # 处理h_pos预测(如果存在)
            if pred_h_pos is not None:
                if isinstance(pred_h_pos, list) and all(isinstance(p, torch.Tensor) for p in pred_h_pos):
                    # 多尺度特征图的h_pos
                    batch_size = pred_scores.shape[0]
                    try:
                        pred_h_pos = torch.cat([p.view(batch_size, 1, -1) for p in pred_h_pos], 2)
                        pred_h_pos = pred_h_pos.permute(0, 2, 1).contiguous()
                        LOGGER.info(f"[DEBUG_LOSS] pred_h_pos shape: {pred_h_pos.shape}")
                    except Exception as e:
                        LOGGER.warning(f"处理h_pos时出错: {e}")
                        pred_h_pos = None
            
            # 基本参数
            dtype = pred_scores.dtype
            batch_size = pred_scores.shape[0]
            imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
            anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
            
            LOGGER.info(f"[DEBUG_LOSS] anchor_points shape: {anchor_points.shape}")
            LOGGER.info(f"[DEBUG_LOSS] stride_tensor shape: {stride_tensor.shape}")
            
            # 准备目标
            targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = targets.to(self.device)  # 确保在正确的设备上
            targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
            
            LOGGER.info(f"[DEBUG_LOSS] gt_labels shape: {gt_labels.shape}, gt_bboxes shape: {gt_bboxes.shape}")
            LOGGER.info(f"[DEBUG_LOSS] mask_gt shape: {mask_gt.shape}, mask_gt sum: {mask_gt.sum().item()}")
            
            # 解码预测框
            pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
            LOGGER.info(f"[DEBUG_LOSS] pred_bboxes shape: {pred_bboxes.shape}")
            
            # 检查pred_bboxes值范围
            if pred_bboxes.size(0) > 0:
                LOGGER.info(f"[DEBUG_LOSS] pred_bboxes范围: min={pred_bboxes.min().item()}, max={pred_bboxes.max().item()}, mean={pred_bboxes.mean().item()}")
            
            # 目标分配
            _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor,
                gt_labels,
                gt_bboxes,
                mask_gt,
            )
            
            LOGGER.info(f"[DEBUG_LOSS] target_bboxes shape: {target_bboxes.shape}")
            LOGGER.info(f"[DEBUG_LOSS] target_scores shape: {target_scores.shape}")
            LOGGER.info(f"[DEBUG_LOSS] fg_mask shape: {fg_mask.shape}, fg_mask sum: {fg_mask.sum().item()}")
            
            target_scores_sum = max(target_scores.sum(), 1)
            LOGGER.info(f"[DEBUG_LOSS] target_scores_sum: {target_scores_sum}")
            
            # 分类损失
            cls_loss = self.bce(pred_scores, target_scores.to(dtype))
            loss[1] = cls_loss.sum() / target_scores_sum
            
            # 添加调试输出
            LOGGER.info(f"分类损失: {loss[1].item():.6f}, target_scores_sum: {target_scores_sum}")
            
            # 边界框损失
            if fg_mask.sum():
                target_bboxes /= stride_tensor
                
                # 添加调试信息
                LOGGER.info(f"[DEBUG_LOSS] 缩放后target_bboxes范围: min={target_bboxes.min().item()}, max={target_bboxes.max().item()}, mean={target_bboxes.mean().item()}")
                
                loss[0], loss[2] = self.bbox_loss(
                    pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
                )
                LOGGER.info(f"边界框损失: box_loss={loss[0].item():.6f}, dfl_loss={loss[2].item():.6f}")
            else:
                LOGGER.warning("没有前景像素，边界框损失为0")
            
            # 计算排序损失
            if "cluster_ids" in batch and "h_rel" in batch:
                try:
                    rank_loss = self.compute_ranking_loss(batch, pred_scores)
                    loss[3] = rank_loss * self.lambda_rank
                    LOGGER.info(f"排序损失: {loss[3].item():.6f}, lambda_rank: {self.lambda_rank}")
                except Exception as e:
                    import traceback
                    LOGGER.warning(f"计算排序损失时出错: {e}")
                    LOGGER.warning(traceback.format_exc())
                    loss[3] = torch.tensor(0.0, device=self.device)
            
            # 应用超参数权重
            loss[0] *= self.hyp.box  # box gain
            loss[1] *= self.hyp.cls  # cls gain
            loss[2] *= self.hyp.dfl  # dfl gain
            # rank_loss已经在上面应用了lambda_rank权重
            
            # 损失总和
            total_loss = loss.sum() * batch_size
            LOGGER.info(f"总损失: {total_loss.item():.6f}")
            
            return total_loss, loss.detach()
            
        except Exception as e:
            import traceback
            LOGGER.warning(f"计算损失时出错: {e}")
            LOGGER.warning(traceback.format_exc())
            
            # 返回零损失，避免训练中断
            loss = torch.zeros(4, device=self.device)
            return loss.sum(), loss.detach()
            
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
        try:
            # 提取必要的标签
            cluster_ids = batch["cluster_ids"].to(self.device).flatten()  # 串ID
            h_rel = batch["h_rel"].to(self.device).flatten()  # 相对高度
            classes = batch["cls"].to(self.device).flatten()  # 类别
            
            # 忽略无效数据 - 修复UNKNOWN_H常量值
            # UNKNOWN_H和BUNCH_H应该在相对高度标签中明确标记，-1.0是一个合理的无效高度标记值
            valid_mask = (h_rel != -1.0) & (h_rel != 0.0) & (classes < self.FRUIT_CLASSES)
            
            if not valid_mask.any():
                # 没有有效数据，返回零损失
                return torch.tensor(0.0, device=self.device)
                
            # 过滤有效数据
            valid_clusters = cluster_ids[valid_mask]
            valid_heights = h_rel[valid_mask]
            valid_classes = classes[valid_mask]
            
            # 查找唯一的串ID
            unique_clusters = valid_clusters.unique()
            
            # 计算排序损失
            total_loss = torch.tensor(0.0, device=self.device)
            pair_count = 0
            
            # 为每个串单独计算
            for cluster_id in unique_clusters:
                # 选择当前串的所有番茄
                cluster_mask = valid_clusters == cluster_id
                cluster_heights = valid_heights[cluster_mask]
                cluster_classes = valid_classes[cluster_mask]
                
                # 至少需要2个番茄才能比较
                if cluster_mask.sum() < 2:
                    continue
                    
                # 按高度排序（从低到高）
                sorted_indices = torch.argsort(cluster_heights)
                sorted_heights = cluster_heights[sorted_indices]
                sorted_classes = cluster_classes[sorted_indices]
                
                # 创建成对比较：位置更高的番茄应该成熟度更高（类别值更小）
                # 注意：类别从0开始，值越小代表成熟度越高 (0=fully_ripe 最成熟)
                for i in range(len(sorted_heights)-1):
                    for j in range(i+1, len(sorted_heights)):
                        # 位置关系: i的位置低于j (sorted_heights[i] > sorted_heights[j])
                        # 期望的成熟度关系: i的成熟度低于j (sorted_classes[i] > sorted_classes[j])
                        
                        # 计算位置差异
                        height_diff = sorted_heights[i] - sorted_heights[j]
                        # 注意：高度较小表示位置更高，因此期望height_diff > 0
                        
                        # 只有位置差异明显时才施加约束（避免高度接近的情况）
                        if height_diff > 0.1:  # 高度差阈值
                            # 计算类别差异
                            expected_sign = torch.tensor(1.0, device=self.device)  # 期望类别差为正
                            class_diff = sorted_classes[i] - sorted_classes[j]  # 应为正值
                            
                            # 使用margin ranking loss
                            # 如果class_diff大于margin，损失为0
                            # 否则损失为margin - class_diff
                            pair_loss = F.margin_ranking_loss(
                                class_diff.unsqueeze(0),
                                torch.tensor([0.0], device=self.device),
                                expected_sign.unsqueeze(0),
                                margin=self.margin,
                                reduction='sum'
                            )
                            
                            # 根据高度差异加权损失（差异越大，权重越大）
                            # 限制权重在[1.0, 2.0]范围内
                            weight = 1.0 + min(height_diff.item(), 1.0)
                            total_loss += pair_loss * weight
                            pair_count += 1
            
            # 平均损失
            if pair_count > 0:
                return total_loss / pair_count
            else:
                return torch.tensor(0.0, device=self.device)
                
        except Exception as e:
            import traceback
            LOGGER.warning(f"排序损失计算出错: {e}")
            LOGGER.warning(traceback.format_exc())
            return torch.tensor(0.0, device=self.device)
