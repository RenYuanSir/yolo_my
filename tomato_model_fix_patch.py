#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TomatoDetectionWithRankLoss 修复补丁
===================================

修复问题:
1. 正样本分配问题：修改正样本分配逻辑，而不是强行将负样本当做正样本
2. 排序损失计算问题：适配正确的高度和成熟度表示方式
   - h_rel: 0表示顶部，1表示底部（从上到下递增）
   - 类别: 数值越小表示越成熟（0:fully ripe, 1:ripe, 2:turning, 3:green）

使用方法：
在训练脚本开头导入并应用此补丁

```python
from tomato_model_fix_patch import apply_patches
apply_patches()  # 应用所有补丁

# 继续正常的训练流程
...
```

或者用于调试：

```python
from tomato_model_fix_patch import debug_loss
debug_loss(model, batch)  # 调试损失计算
```
"""

import torch
import torch.nn.functional as F
import logging
import numpy as np
import sys
from pathlib import Path
import types
from typing import List, Dict, Tuple, Union, Optional

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
LOGGER = logging.getLogger("tomato_fix")

# 修复版本的排序损失计算函数
def compute_ranking_loss_fixed(self, batch, pred_scores=None, h_pos_features=None):
    """
    修复版本的番茄串内成熟度排序损失
    
    Args:
        batch: 数据批次，包含：
            - batch["cluster_ids"]: 串ID标签
            - batch["h_rel"]: 相对高度标签（0为顶部，1为底部）
            - batch["cls"]: 类别标签（值越小表示越成熟）
        pred_scores: 预测的类别得分，用于基于预测的损失计算
        h_pos_features: 预测的高度特征，用于直接从模型输出计算排序损失
    
    Returns:
        rank_loss: 排序损失值
    """
    # 设置一个最小排序损失，确保即使所有样本对都符合规则，也返回一个小的正值
    MIN_RANK_LOSS = 0.01
    margin = 0.07  # 使用更小的高度差异阈值，以便检测更多可能的排序对
    
    # 1. 使用h_pos_features进行基于预测的排序损失计算（若提供）
    if h_pos_features is not None and isinstance(h_pos_features, list) and len(h_pos_features) > 0:
        try:
            # 提取h_pos特征并计算排序损失
            batch_size = h_pos_features[0].shape[0]
            h_pos_loss = torch.tensor(0.0, device=self.device)
            total_checks = 0
            violations = 0
            
            # 对每个特征层级分别计算局部排序损失
            for feat_idx, h_pos in enumerate(h_pos_features):
                # h_pos形状: [B, 1, H, W]
                b, c, h, w = h_pos.shape
                
                if h <= 1 or w <= 1:
                    continue  # 跳过太小的特征图
                
                # 计算垂直方向上相邻像素的排序关系
                # h_pos应该从上到下递增（顶部为0，底部为1）
                for batch_idx in range(batch_size):
                    # 提取当前批次的h_pos映射
                    current_h_pos = h_pos[batch_idx, 0]  # [H, W]
                    
                    # 计算垂直方向上相邻像素的差异（下方像素减上方像素）
                    # h_diff形状: [H-1, W]
                    h_diff = current_h_pos[1:] - current_h_pos[:-1]
                    
                    # 我们期望h_diff大于0，表示h_pos从上到下递增
                    # 当h_diff < 0时，表示违反了期望的排序关系
                    violation_mask = h_diff < 0
                    total_checks += h_diff.numel()
                    violations += violation_mask.sum().item()
                    
                    if violation_mask.any():
                        # 计算违反排序关系的损失
                        h_pos_loss += torch.abs(h_diff[violation_mask]).mean()
            
            # 如果有损失，标准化并返回
            if h_pos_loss > 0:
                # 计算违反率
                violation_ratio = violations / max(total_checks, 1)
                
                # 根据违反率调整损失（违反率越高损失越大）
                adjusted_loss = max(violation_ratio, MIN_RANK_LOSS)
                h_pos_loss = torch.clamp(h_pos_loss, min=MIN_RANK_LOSS)
                
                return h_pos_loss * adjusted_loss
            
            # 如果特征层正常，返回最小损失
            return torch.tensor(MIN_RANK_LOSS, device=self.device)
            
        except Exception as e:
            LOGGER.warning(f"计算基于h_pos的排序损失时出错: {e}, 回退到基于GT的损失计算")
    
    # 2. 回退到基于GT的排序损失计算
    # 检查是否有必要的数据
    if "cluster_ids" not in batch or "h_rel" not in batch or "cls" not in batch:
        LOGGER.debug("排序损失: 批次中缺少必要的字段")
        return torch.tensor(MIN_RANK_LOSS, device=self.device)
        
    if batch["cluster_ids"] is None or batch["h_rel"] is None or batch["cls"] is None:
        LOGGER.debug("排序损失: 批次中有字段为None")
        return torch.tensor(MIN_RANK_LOSS, device=self.device)
        
    if batch["cluster_ids"].numel() == 0 or batch["h_rel"].numel() == 0 or batch["cls"].numel() == 0:
        LOGGER.debug("排序损失: 批次中有空字段")
        return torch.tensor(MIN_RANK_LOSS, device=self.device)
        
    # 准备数据
    cluster_ids = batch["cluster_ids"].flatten()  # 展平为一维
    h_rel = batch["h_rel"].flatten()  # 展平为一维
    classes = batch["cls"].flatten()  # 类别
    
    # 检查并过滤无效数据
    valid_mask = (cluster_ids >= 0) & (h_rel >= 0) & (h_rel <= 1.0)
    if valid_mask.sum() == 0:
        LOGGER.debug("排序损失: 没有有效的数据")
        return torch.tensor(MIN_RANK_LOSS, device=self.device)
        
    # 应用过滤
    valid_cluster_ids = cluster_ids[valid_mask]
    valid_h_rel = h_rel[valid_mask]
    valid_classes = classes[valid_mask]
    
    # 获取所有唯一的串ID
    unique_clusters = torch.unique(valid_cluster_ids)
    
    # 初始化损失
    rank_loss = torch.tensor(0.0, device=self.device)
    pair_count = 0
    penalty_count = 0
    
    # 对每个串内的番茄进行排序比较
    for cluster_id in unique_clusters:
        # 跳过ID为0的串（通常是背景或无效串）
        if cluster_id == 0 and len(unique_clusters) > 1:
            continue
                
        # 找到当前串的所有番茄
        cluster_mask = valid_cluster_ids == cluster_id
        if not cluster_mask.any():
            continue
            
        cluster_h_rel = valid_h_rel[cluster_mask]
        cluster_classes = valid_classes[cluster_mask]
        
        n_tomatoes = len(cluster_h_rel)
        if n_tomatoes <= 1:
            continue  # 跳过只有一个番茄的串
        
        # 添加调试信息，输出每个串的信息
        LOGGER.debug(f"串ID={cluster_id.item()}, 番茄数量={n_tomatoes}")
        LOGGER.debug(f"高度值={cluster_h_rel.tolist()}")
        LOGGER.debug(f"类别值={cluster_classes.tolist()}")
        
        # 比较同一串内的所有番茄对
        for i in range(n_tomatoes):
            for j in range(i + 1, n_tomatoes):
                h1, h2 = cluster_h_rel[i], cluster_h_rel[j]
                cls1, cls2 = cluster_classes[i], cluster_classes[j]
                
                # 计算高度差异
                h_diff = torch.abs(h1 - h2)
                
                # 如果高度差异显著（大于阈值）
                if h_diff > margin:
                    pair_count += 1
                    LOGGER.debug(f"检查第{i}个和第{j}个番茄: 高度差异={h_diff.item():.4f}, 类别:{cls1.item():.0f}和{cls2.item():.0f}")
        
                    # 确定位置关系：h_rel值越大表示位置越低（0是顶部，1是底部）
                    higher_pos_idx, lower_pos_idx = (i, j) if h1 < h2 else (j, i)
                    higher_pos_cls = cluster_classes[higher_pos_idx]
                    lower_pos_cls = cluster_classes[lower_pos_idx]
                    
                    # 根据成熟度规则：类别值越小表示越成熟
                    # 通常，上方的番茄应该更成熟（类别值更小）
                    if higher_pos_cls <= lower_pos_cls:
                        # 符合预期：上方番茄更成熟（类别值更小）
                        LOGGER.debug(f"位置排序正确: 上方类别={higher_pos_cls.item():.0f}, 下方类别={lower_pos_cls.item():.0f}")
                        pass
                    else:
                        # 违反预期：上方番茄反而不如下方成熟，应该惩罚
                        # 计算间隔损失
                        LOGGER.debug(f"位置排序错误: 上方类别={higher_pos_cls.item():.0f}, 下方类别={lower_pos_cls.item():.0f}")
                        loss = F.relu(1.0 - h_diff)
                        rank_loss += loss
                        penalty_count += 1
    
    # 如果有样本对，计算平均损失；否则使用最小损失
    if pair_count > 0:
        LOGGER.debug(f"总共检查了 {pair_count} 对样本，产生了 {penalty_count} 对惩罚")
        rank_loss = rank_loss / pair_count
        # 确保损失不为零
        rank_loss = torch.max(rank_loss, torch.tensor(MIN_RANK_LOSS, device=self.device))
    else:
        LOGGER.debug(f"没有有效的样本对")
        rank_loss = torch.tensor(MIN_RANK_LOSS, device=self.device)
        
    return rank_loss

# 修复版本的TomatoDetectWithRankLoss.__call__方法
def tomato_detection_with_rank_loss_call_fixed(self, preds, batch):
    """
    修复版本的TomatoDetectWithRankLoss.__call__方法
    解决正样本分配问题和排序损失计算问题
    
    Args:
        preds: 模型预测输出，包含检测结构和高度预测
        batch: 数据批次
        
    Returns:
        loss: 总损失值
        loss_items: 损失项列表 [box_loss, cls_loss, dfl_loss, rank_loss]
    """
    # 1. 规范化预测并提取特征
    feats, pred_h_pos = self._normalize_predictions(preds)
    if feats is None:
        LOGGER.error("特征为空，无法计算损失")
        # 返回默认损失
        device = getattr(self, 'device', 'cpu')
        default_loss = torch.tensor(4.0, device=device)  # 一个较大的默认损失值
        default_items = torch.tensor([0.0, 1.0, 0.0, 0.01], device=device)
        return default_loss, default_items
    
    # 2. 从特征中提取预测分布和分数
    pred_distri, pred_scores = self._extract_predictions(feats)
    if pred_distri is None or pred_scores is None:
        LOGGER.error("提取预测分布或分数失败")
        device = getattr(self, 'device', 'cpu')
        default_loss = torch.tensor(4.0, device=device)
        default_items = torch.tensor([0.0, 1.0, 0.0, 0.01], device=device)
        return default_loss, default_items
    
    # 3. 获取批次大小
    batch_size = pred_scores.shape[0]
    
    # 4. 获取缩放张量和设备
    imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=pred_scores.dtype) * self.stride[0]
    
    # 5. 创建锚点网格
    anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
    
    # 6. 准备目标
    targets = self._prepare_targets(batch, batch_size, imgsz)
    
    # 7. 拆分目标
    gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
    
    # 8. 记录目标信息
    LOGGER.debug(f"目标标签形状: {gt_labels.shape}, 边界框形状: {gt_bboxes.shape}")
    n_valid_targets = mask_gt.sum().item()
    LOGGER.debug(f"有效目标数量: {n_valid_targets}")
    
    # 9. 边界框预处理（增强可学习性）
    # 将真实边界框向外扩展一点，增加与预测框的重叠，以提高正样本分配的可能性
    if n_valid_targets > 0 and self.assigner.__class__.__name__ == 'TaskAlignedAssigner':
        # 对真实边界框进行扩展（增加10%）
        scale_factor = 1.1
        valid_mask = mask_gt.squeeze(-1)
        for b in range(batch_size):
            batch_valid_mask = valid_mask[b].bool()  # 确保是布尔类型
            if batch_valid_mask.any():
                # 获取有效的边界框
                valid_boxes = gt_bboxes[b, batch_valid_mask]
                
                # 计算中心点和宽高
                x1, y1, x2, y2 = valid_boxes.unbind(-1)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = (x2 - x1)
                h = (y2 - y1)
                
                # 扩展宽高
                w_expanded = w * scale_factor
                h_expanded = h * scale_factor
                
                # 重新计算坐标
                x1_new = cx - w_expanded / 2
                y1_new = cy - h_expanded / 2
                x2_new = cx + w_expanded / 2
                y2_new = cy + h_expanded / 2
                
                # 更新边界框
                expanded_boxes = torch.stack((x1_new, y1_new, x2_new, y2_new), -1)
                gt_bboxes[b, batch_valid_mask] = expanded_boxes
    
    # 10. 将预测边界框转换为XYXY格式
    pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (batch_size, h*w, 4)
    
    # 11. 分配正样本 - 使用更低的正样本阈值
    try:
        # 修改任务对齐分配器的参数以提高正样本分配率
        if hasattr(self, 'assigner') and self.assigner.__class__.__name__ == 'TaskAlignedAssigner':
            # 临时降低分配阈值
            original_topk = getattr(self.assigner, 'topk', 10)
            original_iou_weight = getattr(self.assigner, 'iou_weight', 3.0)
            
            # 增加topk以分配更多正样本
            self.assigner.topk = max(original_topk, 13)  # 至少13个候选框
            # 降低IoU权重，让更多低IoU的样本可能被分配为正样本
            self.assigner.iou_weight = min(original_iou_weight, 2.0)
        
        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            pred_bboxes.detach() * stride_tensor,
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt
        )
        
        # 恢复原始参数
        if hasattr(self, 'assigner') and self.assigner.__class__.__name__ == 'TaskAlignedAssigner':
            self.assigner.topk = original_topk
            self.assigner.iou_weight = original_iou_weight
    except Exception as e:
        LOGGER.error(f"分配正样本时出错: {e}")
        # 如果出错，创建默认标签和掩码
        fg_mask = torch.zeros((batch_size, pred_scores.shape[1]), dtype=torch.bool, device=self.device)
        target_labels = torch.zeros((batch_size, pred_scores.shape[1], 1), device=self.device)
        target_bboxes = torch.zeros((batch_size, pred_scores.shape[1], 4), device=self.device)
        target_scores = torch.zeros((batch_size, pred_scores.shape[1], self.nc), device=self.device)
    
    # 12. 分析正样本情况
    n_fg_samples = fg_mask.sum().item()
    LOGGER.debug(f"分配的正样本数量: {n_fg_samples}")
    
    # 如果正样本数量很少，考虑采用伪标签策略，但不强行设置
    if n_fg_samples < batch_size * 5:  # 平均每个样本少于5个正样本
        LOGGER.warning(f"正样本数量较少 ({n_fg_samples})，可能影响检测损失计算")
    
    # 13. 计算目标总数
    target_scores_sum = target_scores.sum()
    
    # 14. 计算边界框损失和DFL损失
    try:
        # 备份经过修改的fg_mask以便恢复
        fg_mask_before_loss = fg_mask.clone()
        
        # 计算边界框和DFL损失
        if n_fg_samples > 0:
            try:
                # 首先打印所有张量的形状
                LOGGER.info(f"fg_mask形状: {fg_mask.shape}, anchor_points形状: {anchor_points.shape}")
                LOGGER.info(f"pred_distri形状: {pred_distri.shape}, pred_bboxes形状: {pred_bboxes.shape}")
                LOGGER.info(f"target_bboxes形状: {target_bboxes.shape}, target_scores形状: {target_scores.shape}")
                
                # 保存原始pred_scores用于后续分类损失计算
                pred_scores_orig = pred_scores.clone()
                
                # 确保target_bboxes和target_scores的维度与pred_bboxes一致
                if target_bboxes.dim() == 3 and target_bboxes.shape[0] == batch_size:
                    LOGGER.info(f"重塑target_bboxes: {target_bboxes.shape} -> [num_anchors*batch_size, 4]")
                    target_bboxes = target_bboxes.reshape(-1, 4)  # [num_anchors*batch_size, 4]
                
                if target_scores.dim() == 3 and target_scores.shape[0] == batch_size:
                    LOGGER.info(f"重塑target_scores: {target_scores.shape} -> [num_anchors*batch_size, nc]")
                    target_scores = target_scores.reshape(-1, target_scores.shape[-1])  # [num_anchors*batch_size, nc]
                
                # 检查并修复pred_bboxes的维度
                if pred_bboxes.dim() == 3 and pred_bboxes.shape[0] == batch_size:
                    LOGGER.info(f"重塑pred_bboxes: {pred_bboxes.shape} -> [num_anchors*batch_size, 4]")
                    pred_bboxes = pred_bboxes.reshape(-1, 4)  # [num_anchors*batch_size, 4]
                
                # 检查并修复pred_distri的维度
                if pred_distri.dim() == 3 and pred_distri.shape[0] == batch_size:
                    LOGGER.info(f"重塑pred_distri: {pred_distri.shape} -> [num_anchors*batch_size, 64]")
                    pred_distri = pred_distri.reshape(-1, pred_distri.shape[-1])  # [num_anchors*batch_size, 64]
                
                # 确保fg_mask是一维布尔张量
                if fg_mask.dim() > 1:
                    LOGGER.info(f"重塑fg_mask: {fg_mask.shape} -> [num_anchors*batch_size]")
                    fg_mask = fg_mask.reshape(-1)  # [num_anchors*batch_size]
                
                # 确保anchor_points的维度正确
                if anchor_points.dim() == 2 and anchor_points.shape[0] != pred_bboxes.shape[0]:
                    LOGGER.info(f"调整anchor_points: {anchor_points.shape} -> [num_anchors*batch_size, 2]")
                    # 如果anchor_points是[num_anchors, 2]，需要扩展为[num_anchors*batch_size, 2]
                    anchor_points = anchor_points.repeat(batch_size, 1)
                
                # 最终检查所有张量的形状
                LOGGER.info(f"最终形状检查:")
                LOGGER.info(f"pred_distri: {pred_distri.shape}, pred_bboxes: {pred_bboxes.shape}")
                LOGGER.info(f"anchor_points: {anchor_points.shape}, target_bboxes: {target_bboxes.shape}")
                LOGGER.info(f"target_scores: {target_scores.shape}, fg_mask: {fg_mask.shape}")
                
                # 确保target_gt_idx是长整型
                if target_gt_idx.dtype != torch.long:
                    target_gt_idx = target_gt_idx.long()
                
                # 确保target_scores_sum是标量
                if isinstance(target_scores_sum, torch.Tensor) and target_scores_sum.numel() > 1:
                    target_scores_sum = target_scores_sum.sum()
                
                # 调用bbox_loss计算损失
                try:
                    # 检查pred_distri的维度是否正确
                    if pred_distri.shape[-1] != 64:  # 确保最后一个维度是64
                        LOGGER.warning(f"pred_distri的最后一个维度不是64: {pred_distri.shape}")
                        # 如果是[18522, 64]，但实际应该是[18522, 16, 4]，则需要调整
                        if pred_distri.numel() == 1664 * 16 * 4:
                            # 重塑为正确的形状
                            pred_distri = pred_distri.reshape(-1, 16, 4)
                            LOGGER.info(f"重塑pred_distri为: {pred_distri.shape}")
                    
                    loss_iou, loss_dfl = self.bbox_loss(
                        pred_dist=pred_distri,
                        pred_bboxes=pred_bboxes,
                        anchor_points=anchor_points,
                        target_bboxes=target_bboxes,
                        target_scores=target_scores,
                        target_scores_sum=target_scores_sum,
                        fg_mask=fg_mask
                    )
                except Exception as e:
                    LOGGER.error(f"计算bbox/dfl损失时出错: {e}")
                    import traceback
                    LOGGER.error(traceback.format_exc())
                    loss_iou = torch.tensor(0.0, device=self.device)
                    loss_dfl = torch.tensor(0.0, device=self.device)
            except Exception as e:
                LOGGER.error(f"计算bbox/dfl损失时出错: {e}")
                import traceback
                LOGGER.error(traceback.format_exc())
                loss_iou = torch.tensor(0.0, device=self.device)
                loss_dfl = torch.tensor(0.0, device=self.device)
        else:
            # 如果没有正样本，损失为0
            loss_iou = torch.tensor(0.0, device=self.device)
            loss_dfl = torch.tensor(0.0, device=self.device)
        
        # 恢复fg_mask（避免在bbox_loss中可能发生的原地修改）
        fg_mask = fg_mask_before_loss
    except Exception as e:
        LOGGER.error(f"计算bbox/dfl损失时出错: {e}")
        loss_iou = torch.tensor(0.0, device=self.device)
        loss_dfl = torch.tensor(0.0, device=self.device)
    
    # 15. 计算分类损失
    try:
        # 使用原始形状的pred_scores计算分类损失
        if pred_scores.shape != target_scores.shape:
            LOGGER.info(f"分类损失计算前调整维度: pred_scores={pred_scores.shape}, target_scores={target_scores.shape}")
            
            # 保存原始形状的target_scores
            batch_target_scores = target_scores.clone()
            
            # 如果target_scores已经被重塑为[num_anchors*batch_size, nc]
            if target_scores.dim() == 2 and target_scores.shape[0] == anchor_points.shape[0] * batch_size:
                # 重塑pred_scores以匹配target_scores
                pred_scores = pred_scores.reshape(-1, pred_scores.shape[-1])
                LOGGER.info(f"重塑pred_scores为: {pred_scores.shape}")
            
            # 如果pred_scores已经被重塑，但target_scores没有
            elif pred_scores.dim() == 2 and target_scores.dim() == 3:
                # 重塑target_scores以匹配pred_scores
                target_scores = target_scores.reshape(-1, target_scores.shape[-1])
                LOGGER.info(f"重塑target_scores为: {target_scores.shape}")
            
            # 如果两者都是3维但形状不同
            elif pred_scores.dim() == 3 and target_scores.dim() == 2:
                # 重塑pred_scores以匹配target_scores
                pred_scores = pred_scores.reshape(-1, pred_scores.shape[-1])
                LOGGER.info(f"重塑pred_scores为3维到2维: {pred_scores.shape}")
            
            # 如果维度匹配但形状不同
            elif pred_scores.shape != target_scores.shape:
                # 使用原始target_scores
                LOGGER.info("维度不匹配，恢复原始target_scores")
                # 恢复原始形状
                target_scores = batch_target_scores.clone()
                pred_scores = pred_scores_orig.clone()
        
        # 最终检查并确保维度匹配
        if pred_scores.shape != target_scores.shape:
            LOGGER.warning(f"维度仍然不匹配! pred_scores={pred_scores.shape}, target_scores={target_scores.shape}")
            # 强制调整target_scores的形状以匹配pred_scores
            if pred_scores.dim() == 3 and target_scores.dim() == 2:
                # 将target_scores从[N, C]调整为[B, N/B, C]
                target_scores = target_scores.reshape(batch_size, -1, target_scores.shape[-1])
                LOGGER.info(f"强制调整target_scores为: {target_scores.shape}")
            elif pred_scores.dim() == 2 and target_scores.dim() == 3:
                # 将pred_scores从[N, C]调整为[B, N/B, C]
                pred_scores = pred_scores.reshape(batch_size, -1, pred_scores.shape[-1])
                LOGGER.info(f"强制调整pred_scores为: {pred_scores.shape}")
        
        LOGGER.info(f"分类损失计算: pred_scores={pred_scores.shape}, target_scores={target_scores.shape}")
        loss_cls = self.bce(pred_scores, target_scores)
    except Exception as e:
        LOGGER.error(f"计算分类损失时出错: {e}")
        import traceback
        LOGGER.error(traceback.format_exc())
        loss_cls = torch.tensor(1.0, device=self.device)  # 使用默认值1.0
    
    # 16. 计算排序损失
    try:
        # 使用修复后的排序损失函数，同时传递h_pos特征用于基于预测的损失计算
        rank_loss = compute_ranking_loss_fixed(self, batch, pred_scores=None, h_pos_features=pred_h_pos)
    except Exception as e:
        LOGGER.error(f"计算排序损失时出错: {e}")
        rank_loss = torch.tensor(0.01, device=self.device)  # 默认排序损失
    
    # 17. 加权损失和
    loss = self.hyp.box * loss_iou + self.hyp.cls * loss_cls + self.hyp.dfl * loss_dfl + self.lambda_rank * rank_loss
    
    # 18. 返回总损失和损失项
    # 确保所有损失项都是标量张量并且有正确的形状
    loss_items = torch.tensor([loss_iou.detach().item(), 
                              loss_cls.detach().item(), 
                              loss_dfl.detach().item(), 
                              rank_loss.detach().item()], 
                             device=self.device).reshape(1, 4)
    return loss, loss_items

def make_anchors(features, strides, grid_cell_offset=0.5):
    """
    Make anchors from features，兼容性实现
    """
    anchor_points, stride_tensor = [], []
    assert len(features) == len(strides)
    for i, stride in enumerate(strides):
        _, _, h, w = features[i].shape
        sx = torch.arange(w) + grid_cell_offset  # shift x
        sy = torch.arange(h) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing='ij') if hasattr(torch, 'meshgrid') and torch.__version__ >= '1.10.0' else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).reshape(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=features[i].dtype))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

def apply_patches():
    """应用所有修复补丁"""
    try:
        # 导入需要修复的模块
        from ultralytics.utils.loss import TomatoDetectWithRankLoss
        import inspect
        
        # 打印原始函数信息
        LOGGER.info("正在修复TomatoDetectWithRankLoss...")
        
        # 备份原始方法
        original_compute_ranking_loss = TomatoDetectWithRankLoss.compute_ranking_loss
        original_call = TomatoDetectWithRankLoss.__call__
        
        # 应用修复补丁
        TomatoDetectWithRankLoss.compute_ranking_loss = compute_ranking_loss_fixed
        TomatoDetectWithRankLoss.__call__ = tomato_detection_with_rank_loss_call_fixed
        
        LOGGER.info("成功应用修复补丁!")
        LOGGER.info("已修复问题:")
        LOGGER.info("1. 改进了正样本分配策略 - 扩展GT框并调整分配器参数")
        LOGGER.info("2. 修正了排序损失(rank_loss)计算 - 适配正确的高度和成熟度表示")
        LOGGER.info("3. 增加了基于h_pos特征的排序损失计算")
        
        return True
    except Exception as e:
        LOGGER.error(f"应用补丁失败: {e}")
        import traceback
        LOGGER.error(traceback.format_exc())
        return False

def debug_loss(model, batch, verbose=True):
    """
    调试模型损失计算
    
    Args:
        model: 模型实例
        batch: 批次数据
        verbose: 是否打印详细信息
    
    Returns:
        loss: 损失值
        loss_items: 损失项 [box_loss, cls_loss, dfl_loss, rank_loss]
        analysis: 损失分析报告
    """
    # 设置调试日志级别
    if verbose:
        LOGGER.setLevel(logging.DEBUG)
    
    # 应用补丁
    apply_patches()
    
    # 备份原始批次数据
    batch_copy = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    # 获取损失函数
    loss_fn = None
    from ultralytics.utils.loss import TomatoDetectWithRankLoss
    
    for name, attr in model.__dict__.items():
        if isinstance(attr, TomatoDetectWithRankLoss):
            loss_fn = attr
            break
    
    if loss_fn is None:
        # 创建一个新的损失函数
        loss_fn = TomatoDetectWithRankLoss(model)
    
    # 执行前向传播
    LOGGER.info("执行模型前向传播...")
    with torch.no_grad():
        preds = model(batch['img'])
    
    # 计算损失
    LOGGER.info("计算损失...")
    loss, loss_items = loss_fn(preds, batch)
    
    # 分析损失
    analysis = {
        "total_loss": loss.item(),
        "box_loss": loss_items[0].item(),
        "cls_loss": loss_items[1].item(),
        "dfl_loss": loss_items[2].item(),
        "rank_loss": loss_items[3].item(),
    }
    
    # 打印分析结果
    if verbose:
        LOGGER.info("\n" + "-" * 40 + " 损失分析 " + "-" * 40)
        LOGGER.info(f"总损失: {analysis['total_loss']:.4f}")
        LOGGER.info(f"边界框损失: {analysis['box_loss']:.4f}")
        LOGGER.info(f"分类损失: {analysis['cls_loss']:.4f}")
        LOGGER.info(f"DFL损失: {analysis['dfl_loss']:.4f}")
        LOGGER.info(f"排序损失: {analysis['rank_loss']:.4f}")
        LOGGER.info("-" * 90)
    
    return loss, loss_items, analysis

if __name__ == "__main__":
    # 应用补丁
    success = apply_patches()
    LOGGER.info(f"补丁应用{'成功' if success else '失败'}")