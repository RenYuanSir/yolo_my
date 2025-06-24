#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试TomatoDetectWithRankLoss损失函数
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加ultralytics模块到路径
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ultralytics.utils.loss import TomatoDetectWithRankLoss
from ultralytics.data.dataset import TomatoYOLODataset
from ultralytics.cfg import get_cfg
from torch.utils.data import DataLoader
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_tomato_loss")

def create_mock_model():
    """创建一个模拟的模型用于初始化损失函数"""
    class MockModel:
        def __init__(self):
            self.args = get_cfg()
            self.args.box = 7.5
            self.args.cls = 0.5
            self.args.dfl = 1.5
            
            class MockDetect:
                def __init__(self):
                    self.nc = 6  # 6个类别
                    self.reg_max = 16  # 默认值
                    self.stride = torch.tensor([8., 16., 32.])
                    
            self.model = [MockDetect()]
            self._params = [torch.zeros(1, requires_grad=True, device="cuda" if torch.cuda.is_available() else "cpu")]
            
        def parameters(self):
            """模拟模型参数，返回一个迭代器"""
            for p in self._params:
                yield p
            
    return MockModel()

def create_mock_batch(batch_size=2, device="cpu"):
    """创建一个模拟的批次数据"""
    # 创建图像
    img = torch.rand(batch_size, 3, 640, 640, device=device)
    
    # 创建边界框 (每张图像10个目标)
    n_boxes_per_img = 10
    total_boxes = batch_size * n_boxes_per_img
    
    # 批次索引
    batch_idx = torch.cat([torch.full((n_boxes_per_img,), i, device=device) for i in range(batch_size)])
    
    # 类别 (0-5)
    cls = torch.randint(0, 6, (total_boxes, 1), device=device).float()
    
    # 边界框 (x, y, w, h) - 归一化坐标
    bboxes = torch.rand(total_boxes, 4, device=device)
    bboxes[:, 2:] = bboxes[:, 2:] * 0.3  # 确保宽高合理
    
    # 串ID (1-3, 0表示无串)
    cluster_ids = torch.randint(0, 4, (total_boxes, 1), device=device).float()
    
    # 相对高度 (0.0-1.0, 0表示最高位置)
    h_rel = torch.rand(total_boxes, 1, device=device)
    
    # 组合为批次
    batch = {
        "img": img,
        "batch_idx": batch_idx,
        "cls": cls,
        "bboxes": bboxes,
        "cluster_ids": cluster_ids,
        "h_rel": h_rel,
    }
    
    return batch

def create_mock_predictions(batch_size=2, device="cpu"):
    """创建模拟的预测结果"""
    # 三个特征层 (P3, P4, P5)
    stride_sizes = [8, 16, 32]
    nc = 6  # 类别数
    reg_max = 16  # 每个方向的分布数量
    
    # 计算每个特征层的大小
    feature_sizes = [(640 // s, 640 // s) for s in stride_sizes]
    
    # 创建特征层
    features = []
    for h, w in feature_sizes:
        # 每个位置预测 (4 * reg_max + nc) 个值
        # 4 * reg_max: 四个方向的分布
        # nc: 类别预测
        f = torch.rand(batch_size, 4 * reg_max + nc, h, w, device=device)
        features.append(f)
    
    return features

def test_tomato_loss():
    """测试TomatoDetectWithRankLoss损失函数"""
    logger.info("开始测试TomatoDetectWithRankLoss损失函数")
    
    # 创建设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 创建模拟模型和损失函数
    mock_model = create_mock_model()
    loss_fn = TomatoDetectWithRankLoss(mock_model, lambda_rank=0.2)
    logger.info("创建损失函数成功")
    
    # 创建模拟批次和预测
    batch = create_mock_batch(batch_size=2, device=device)
    predictions = create_mock_predictions(batch_size=2, device=device)
    logger.info(f"创建模拟数据成功: 批次中图像形状 {batch['img'].shape}, 目标总数 {len(batch['cls'])}")
    
    # 计算损失
    try:
        total_loss, loss_items = loss_fn(predictions, batch)
        logger.info(f"损失计算成功:")
        logger.info(f"  总损失: {total_loss.item():.6f}")
        logger.info(f"  边界框损失: {loss_items[0].item():.6f}")
        logger.info(f"  分类损失: {loss_items[1].item():.6f}")
        logger.info(f"  DFL损失: {loss_items[2].item():.6f}")
        logger.info(f"  排序损失: {loss_items[3].item():.6f}")
        return True
    except Exception as e:
        import traceback
        logger.error(f"损失计算失败: {e}")
        logger.error(traceback.format_exc())
        return False

def test_with_real_data():
    """使用真实数据集测试损失函数"""
    logger.info("开始使用真实数据集测试损失函数")
    
    # 设置数据集路径
    dataset_path = Path("/root/shared-nvme/tomato_transStyle/train/images")
    if not dataset_path.exists():
        logger.warning(f"数据集路径不存在: {dataset_path}")
        return False
        
    # 创建设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 创建配置
    args = get_cfg()
    args.task = 'tomato'
    args.imgsz = 640
    args.batch = 4
    args.mosaic = 0.0  # 禁用mosaic增强，简化调试
    
    # 创建数据集
    try:
        dataset = TomatoYOLODataset(
            img_path=dataset_path,
            imgsz=args.imgsz,
            batch_size=args.batch,
            augment=False,  # 禁用数据增强
            hyp=args,
            rect=False,
            cache=None,
            single_cls=False,
            stride=32,
            pad=0.0,
            prefix='test: ',
            data={'path': str(dataset_path.parent), 'nc': 6, 'names': [f'class{i}' for i in range(6)], 'has_cluster_id': True, 'has_h_rel': True},
        )
        logger.info(f"创建数据集成功: {len(dataset)} 张图像")
    except Exception as e:
        import traceback
        logger.error(f"创建数据集失败: {e}")
        logger.error(traceback.format_exc())
        return False
    
    # 创建数据加载器
    try:
        dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, collate_fn=dataset.collate_fn)
        batch = next(iter(dataloader))
        logger.info(f"数据加载器创建成功: 批次大小 {batch['img'].shape[0]}, 目标总数 {len(batch['cls'])}")
    except Exception as e:
        import traceback
        logger.error(f"创建数据加载器失败: {e}")
        logger.error(traceback.format_exc())
        return False
    
    # 创建模拟模型和损失函数
    mock_model = create_mock_model()
    loss_fn = TomatoDetectWithRankLoss(mock_model, lambda_rank=0.2)
    logger.info("创建损失函数成功")
    
    # 创建模拟预测
    predictions = create_mock_predictions(batch_size=batch['img'].shape[0], device=device)
    
    # 将批次数据移动到设备
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    # 计算损失
    try:
        total_loss, loss_items = loss_fn(predictions, batch)
        logger.info(f"损失计算成功:")
        logger.info(f"  总损失: {total_loss.item():.6f}")
        logger.info(f"  边界框损失: {loss_items[0].item():.6f}")
        logger.info(f"  分类损失: {loss_items[1].item():.6f}")
        logger.info(f"  DFL损失: {loss_items[2].item():.6f}")
        logger.info(f"  排序损失: {loss_items[3].item():.6f}")
        return True
    except Exception as e:
        import traceback
        logger.error(f"损失计算失败: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    # 测试模拟数据
    result1 = test_tomato_loss()
    
    # 测试真实数据
    result2 = test_with_real_data()
    
    if result1 and result2:
        logger.info("所有测试通过!")
        sys.exit(0)
    else:
        logger.error("测试失败!")
        sys.exit(1) 