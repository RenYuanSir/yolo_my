# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import numpy as np

from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import ConfusionMatrix


class TomatoValidator(DetectionValidator):
    """
    TomatoValidator类用于验证番茄检测模型，继承自DetectionValidator。
    
    添加了对h_pos预测的评估和排序指标的计算。
    
    Example:
        ```python
        from ultralytics.models.yolo.tomato import TomatoValidator

        args = dict(model="yolov12-tomato.pt", data="tomato.yaml")
        validator = TomatoValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """初始化番茄检测验证器，添加特定于番茄的指标。"""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.h_pos_stats = {'tp': [], 'conf': [], 'pred_h': [], 'target_h': []}
        self.rank_stats = {'correct': 0, 'total': 0}

    def init_metrics(self, model):
        """初始化评估指标。"""
        super().init_metrics(model)
        # 初始化特定于番茄的指标
        self.h_pos_stats = {'tp': [], 'conf': [], 'pred_h': [], 'target_h': []}
        self.rank_stats = {'correct': 0, 'total': 0}
        
    def postprocess(self, preds):
        """对预测结果应用后处理。"""
        # 检查是否是包含h_pos的预测结果
        if isinstance(preds, dict) and 'h_pos' in preds:
            h_pos = preds['h_pos']  # 保存h_pos预测
            preds = preds['features']  # 获取边界框预测
            
            # 后处理边界框预测
            preds = super().postprocess(preds)
            
            # 关联h_pos预测和边界框
            for i, (pred, h) in enumerate(zip(preds, h_pos)):
                if len(pred):
                    # 将h_pos添加到预测结果中
                    h = h.flatten(2).permute(0, 2, 1)  # (bs, n_points, 1)
                    h_idx = torch.zeros_like(pred[:, 0], dtype=torch.long)
                    pred = torch.cat((pred, h[i][h_idx].unsqueeze(-1)), dim=1)
                preds[i] = pred
            
            return preds
        else:
            # 常规后处理
            return super().postprocess(preds)
    
    def update_metrics(self, preds, batch):
        """更新评估指标。"""
        # 调用父类方法处理常规检测指标
        super().update_metrics(preds, batch)
        
        # 处理h_pos特定指标
        for si, pred in enumerate(preds):
            if len(pred) > 0 and pred.shape[1] > 6:  # 确保有h_pos预测
                # 提取h_pos预测和真实值
                h_pos_pred = pred[:, 6]
                
                # 获取匹配的真实框
                pbatch = self._prepare_batch(si, batch)
                cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
                if len(cls) == 0:
                    continue
                    
                # 获取IoU矩阵
                iou = box_iou(bbox, pred[:, :4])
                
                # 匹配检测和真实框
                max_iou, max_idx = iou.max(0)
                matched = max_iou > 0.5
                
                # 记录匹配的h_pos
                if matched.sum() > 0:
                    # 这里假设batch中有h_pos字段，实际需要根据数据结构调整
                    if 'h_pos' in batch:
                        h_pos_true = batch['h_pos'][si][max_idx[matched]]
                        self.h_pos_stats['tp'].append(matched)
                        self.h_pos_stats['conf'].append(pred[matched, 4])
                        self.h_pos_stats['pred_h'].append(h_pos_pred[matched])
                        self.h_pos_stats['target_h'].append(h_pos_true)
    
    def finalize_metrics(self, *args, **kwargs):
        """完成指标计算，添加h_pos评估和排序指标。"""
        # 调用父类方法处理常规检测指标
        metrics = super().finalize_metrics(*args, **kwargs)
        
        # 处理h_pos特定指标
        if self.h_pos_stats['tp']:
            tp = torch.cat(self.h_pos_stats['tp'])
            conf = torch.cat(self.h_pos_stats['conf']) if len(self.h_pos_stats['conf']) else torch.zeros(0)
            pred_h = torch.cat(self.h_pos_stats['pred_h']) if len(self.h_pos_stats['pred_h']) else torch.zeros(0)
            target_h = torch.cat(self.h_pos_stats['target_h']) if len(self.h_pos_stats['target_h']) else torch.zeros(0)
            
            # 计算h_pos的MAE
            if len(pred_h) and len(target_h):
                h_mae = torch.mean(torch.abs(pred_h - target_h)).item()
                metrics['metrics/h_mae'] = h_mae
                LOGGER.info(f"H-pos MAE: {h_mae:.4f}")
                
                # 计算排序正确率
                if len(pred_h) > 1:
                    # 根据真实h_pos排序
                    sorted_indices = torch.argsort(target_h)
                    pred_sorted = pred_h[sorted_indices]
                    
                    # 计算排序是否正确
                    correct_order = 0
                    total_pairs = 0
                    
                    for i in range(len(pred_sorted)-1):
                        for j in range(i+1, len(pred_sorted)):
                            total_pairs += 1
                            if pred_sorted[i] <= pred_sorted[j]:
                                correct_order += 1
                    
                    if total_pairs > 0:
                        rank_acc = correct_order / total_pairs
                        metrics['metrics/rank_acc'] = rank_acc
                        LOGGER.info(f"H-pos Ranking Accuracy: {rank_acc:.4f}")
        
        return metrics
        
    def get_desc(self):
        """返回格式化的字符串，总结YOLO模型的类指标。"""
        return ("%22s" + "%11s" * 7) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95", "H-MAE") 