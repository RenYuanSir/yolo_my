#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试TomatoYOLODataset类的脚本
验证是否能正确加载番茄数据集，特别是cluster_ids和h_rel字段
"""

import os
import sys
import torch
import yaml
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 直接从当前目录导入
from ultralytics.data.dataset import TomatoYOLODataset
from ultralytics.utils import LOGGER

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_tomato_dataset")

def visualize_sample(img, labels, cluster_ids=None, h_rel=None, save_path=None):
    """
    可视化数据样本，包括边界框、cluster_id和h_rel
    
    Args:
        img: 图像张量 (C, H, W)
        labels: 标签张量 (N, 5) - 包含 [cls, x, y, w, h]
        cluster_ids: 串ID张量 (N, 1)
        h_rel: 相对高度张量 (N, 1)
        save_path: 保存路径
    """
    # 转换图像格式
    img = img.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
    img = (img * 255).astype(np.uint8)
    
    # 创建图像
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    
    # 获取图像尺寸
    height, width = img.shape[:2]
    
    # 确保labels是2D张量
    if labels.dim() > 2:
        labels = labels.squeeze()
    
    # 绘制边界框和标签
    for i in range(labels.shape[0]):
        cls, x, y, w, h = labels[i].tolist()
        
        # 转换为像素坐标
        x1 = int((x - w/2) * width)
        y1 = int((y - h/2) * height)
        x2 = int((x + w/2) * width)
        y2 = int((y + h/2) * height)
        
        # 绘制边界框
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
        
        # 准备标签文本
        label_text = f"Class: {int(cls)}"
        
        # 添加cluster_id和h_rel信息（如果有）
        if cluster_ids is not None:
            c_id = cluster_ids[i].item()
            if c_id >= 0:  # 有效的cluster_id
                label_text += f", C:{int(c_id)}"
                
        if h_rel is not None:
            h = h_rel[i].item()
            if h >= 0:  # 有效的h_rel
                label_text += f", H:{h:.2f}"
        
        # 绘制标签文本
        plt.text(x1, y1-5, label_text, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.axis('off')
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def test_tomato_dataset():
    """测试TomatoYOLODataset类"""
    logger.info("开始测试TomatoYOLODataset类")
    
    # 使用永久的配置文件
    yaml_path = Path("tomato_data.yaml")
    if not yaml_path.exists():
        logger.error(f"配置文件不存在: {yaml_path}")
        return False
    
    # 从YAML文件加载数据配置
    with open(yaml_path, 'r',encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    logger.info(f"加载数据配置: {data_config}")
    
    # 检查数据集路径
    dataset_path = Path(data_config['train'])
    if not dataset_path.exists():
        logger.error(f"数据集路径不存在: {dataset_path}")
        return False
    
    # 创建参数
    class Args:
        def __init__(self):
            self.imgsz = 640
            self.batch = 2
            self.rect = False
            self.cache = None
            self.single_cls = False
            self.augment = False
            self.stride = 32
            self.pad = 0.0
            self.loss = "TomatoDetectWithRankLoss"  # 确保启用cluster_ids和h_rel
            # 添加必要的超参数
            self.mosaic = 0.0
            self.mixup = 0.0
            self.copy_paste = 0.0
            self.mask_ratio = 4.0
            self.overlap_mask = True
            self.bgr = 0.0
    
    args = Args()
    
    try:
        # 创建数据集
        dataset = TomatoYOLODataset(
            img_path=dataset_path,
            imgsz=args.imgsz,
            batch_size=args.batch,
            augment=args.augment,
            hyp=args,
            rect=args.rect,
            cache=args.cache,
            single_cls=args.single_cls,
            stride=args.stride,
            pad=args.pad,
            prefix='test: ',
            data=data_config,  # 使用从YAML加载的配置
            args=args,  # 传递args参数，确保TomatoDetectWithRankLoss标志被识别
        )
        logger.info(f"创建数据集成功: {len(dataset)} 张图像")
        
        # 检查数据集属性
        logger.info(f"数据集属性: has_cluster_ids={dataset.has_cluster_ids}, has_h_rel={dataset.has_h_rel}")
        
        # 创建数据加载器
        dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, collate_fn=TomatoYOLODataset.collate_fn)
        logger.info(f"创建数据加载器成功")
        
        # 获取一个批次
        batch = next(iter(dataloader))
        logger.info(f"获取批次成功: 批次大小 {batch['img'].shape[0]}")
        
        # 检查批次中的关键字段
        logger.info(f"批次字段: {list(batch.keys())}")
        
        # 检查cluster_ids和h_rel字段
        if 'cluster_ids' in batch:
            logger.info(f"cluster_ids: 形状={batch['cluster_ids'].shape}, 类型={batch['cluster_ids'].dtype}")
            logger.info(f"cluster_ids样本: {batch['cluster_ids'][:5]}")
        else:
            logger.warning("批次中没有cluster_ids字段")
        
        if 'h_rel' in batch:
            logger.info(f"h_rel: 形状={batch['h_rel'].shape}, 类型={batch['h_rel'].dtype}")
            logger.info(f"h_rel样本: {batch['h_rel'][:5]}")
        else:
            logger.warning("批次中没有h_rel字段")
        
        # 可视化第一个样本
        img = batch['img'][0]
        idx = batch['batch_idx'] == 0
        
        # 确保cls和bboxes的维度匹配
        cls = batch['cls'][idx]
        if cls.dim() == 1:
            cls = cls.unsqueeze(1)
        elif cls.dim() > 2:
            cls = cls.squeeze(1)
        
        bboxes = batch['bboxes'][idx]
        if bboxes.dim() > 2:
            bboxes = bboxes.squeeze(1)
        
        # 组合cls和bboxes
        labels = torch.cat([cls, bboxes], dim=1)
        
        cluster_ids = batch['cluster_ids'][idx] if 'cluster_ids' in batch else None
        h_rel = batch['h_rel'][idx] if 'h_rel' in batch else None
        
        save_dir = Path('runs/test_tomato')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        visualize_sample(
            img, 
            labels, 
            cluster_ids, 
            h_rel, 
            save_path=str(save_dir / 'sample_visualization.jpg')
        )
        logger.info(f"样本可视化已保存至 {save_dir / 'sample_visualization.jpg'}")
        
        return True
        
    except Exception as e:
        import traceback
        logger.error(f"测试失败: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_tomato_dataset()
    if success:
        logger.info("测试成功完成!")
    else:
        logger.error("测试失败!") 