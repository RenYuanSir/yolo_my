#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import numpy as np
import logging
from pathlib import Path

# 设置日志级别
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BBoxNormalizeTest")

# 添加父目录到PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics.utils.loss import TomatoDetectWithRankLoss
from ultralytics.nn.tasks import TomatoDetectionModel

def create_unnormalized_bbox_batch():
    """创建一个包含未归一化边界框的批次"""
    # 创建一个虚拟的边界框批次 - 使用像素坐标 [cx, cy, w, h]
    # 假设图像大小为640x640
    img_size = 640
    
    # 批次中包含3个边界框，格式为[cx, cy, w, h]
    bboxes = torch.tensor([
        [320.0, 240.0, 100.0, 150.0],  # 中心点(320, 240)，宽100，高150
        [420.0, 320.0, 80.0, 120.0],   # 中心点(420, 320)，宽80，高120
        [150.0, 180.0, 60.0, 90.0],    # 中心点(150, 180)，宽60，高90
    ], dtype=torch.float32)
    
    # 类别ID
    cls = torch.tensor([
        [0.0],  # 第一个框是类别0
        [1.0],  # 第二个框是类别1
        [2.0],  # 第三个框是类别2
    ], dtype=torch.float32)
    
    # 批次索引
    batch_idx = torch.tensor([
        [0.0],  # 第一个框在第一张图
        [0.0],  # 第二个框在第一张图
        [1.0],  # 第三个框在第二张图
    ], dtype=torch.float32)
    
    # 串ID
    cluster_ids = torch.tensor([
        [1.0],  # 第一个框属于串1
        [1.0],  # 第二个框属于串1
        [2.0],  # 第三个框属于串2
    ], dtype=torch.float32)
    
    # 相对高度
    h_rel = torch.tensor([
        [0.2],  # 第一个框相对高度0.2
        [0.5],  # 第二个框相对高度0.5
        [0.3],  # 第三个框相对高度0.3
    ], dtype=torch.float32)
    
    # 创建批次字典
    batch = {
        'bboxes': bboxes,
        'cls': cls,
        'batch_idx': batch_idx,
        'cluster_ids': cluster_ids,
        'h_rel': h_rel
    }
    
    return batch, img_size

def test_prepare_targets():
    """测试_prepare_targets方法"""
    logger.info("创建TomatoDetectWithRankLoss实例...")
    
    # 创建一个简单的模型
    model = TomatoDetectionModel()
    
    # 创建TomatoDetectWithRankLoss实例
    loss = TomatoDetectWithRankLoss(model)
    
    # 创建未归一化的边界框批次
    batch, img_size = create_unnormalized_bbox_batch()
    
    logger.info("输入批次:")
    for k, v in batch.items():
        logger.info(f"{k}: 形状={v.shape}, 值范围=[{v.min().item():.4f}, {v.max().item():.4f}]")
        if k == 'bboxes':
            logger.info(f"边界框详情:")
            for i, bbox in enumerate(v):
                logger.info(f"  框{i}: {bbox.tolist()}")
    
    # 手动修改_prepare_targets方法
    orig_prepare_targets = loss._prepare_targets
    
    def patched_prepare_targets(batch, batch_size, imgsz):
        """修补版_prepare_targets方法，增加额外日志"""
        logger.info("=== 调用_prepare_targets方法 ===")
        logger.info(f"batch_size={batch_size}, imgsz={imgsz}")
        
        try:
            # 检查边界框是否需要归一化
            if batch['bboxes'].numel() > 0:
                is_normalized = True  # 假设边界框已标准化
                if (batch['bboxes'] > 1.0).any():
                    logger.info(f"检测到边界框坐标 > 1.0，最大值: {batch['bboxes'].max().item()}")
                    is_normalized = False
                
                # 如果边界框未标准化，尝试归一化它们
                if not is_normalized:
                    logger.info("尝试标准化边界框坐标...")
                    # 获取图像尺寸
                    img_w = img_h = img_size
                    
                    # 获取边界框数据
                    bbox_data = batch['bboxes']
                    
                    # 所有标签都是XYWH格式 [cx, cy, w, h]
                    logger.info(f"对XYWH格式边界框进行归一化，原始范围: [{bbox_data.min().item():.1f}, {bbox_data.max().item():.1f}]")
                    
                    # 分别归一化中心点和宽高
                    cx, cy, w, h = bbox_data.unbind(-1)
                    cx = cx / img_w
                    cy = cy / img_h
                    w = w / img_w
                    h = h / img_h
                    
                    # 确保中心点在有效范围内
                    cx = torch.clamp(cx, min=0.001, max=0.999)
                    cy = torch.clamp(cy, min=0.001, max=0.999)
                    
                    # 确保宽高合理（在[0.001, 1.0]范围内）
                    w = torch.clamp(w, min=0.001, max=0.999)
                    h = torch.clamp(h, min=0.001, max=0.999)
                    
                    # 整合归一化后的边界框
                    batch['bboxes'] = torch.stack([cx, cy, w, h], dim=1)
                    logger.info(f"归一化完成，新边界框范围: cx=[{cx.min().item():.4f}, {cx.max().item():.4f}], cy=[{cy.min().item():.4f}, {cy.max().item():.4f}]")
                    logger.info(f"归一化完成，新边界框范围: w=[{w.min().item():.4f}, {w.max().item():.4f}], h=[{h.min().item():.4f}, {h.max().item():.4f}]")
            
            # 调用原始方法
            result = orig_prepare_targets(batch, batch_size, imgsz)
            
            # 检查结果
            if result is not None:
                logger.info(f"处理结果: 形状={result.shape}, 值范围=[{result.min().item() if result.numel() > 0 else 'N/A'}, {result.max().item() if result.numel() > 0 else 'N/A'}]")
                if result.numel() > 0:
                    logger.info("前几个目标的值:")
                    for i in range(min(3, result.shape[0])):
                        logger.info(f"  目标{i}: {result[i, 0].tolist() if result.shape[1] > 0 else []}")
            
            return result
        except Exception as e:
            logger.error(f"处理失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    # 替换方法
    loss._prepare_targets = patched_prepare_targets
    
    # 调用_prepare_targets方法
    batch_size = 2  # 我们的batch包含2张图片
    imgsz = torch.tensor([img_size, img_size])
    
    logger.info("调用_prepare_targets方法...")
    result = loss._prepare_targets(batch, batch_size, imgsz)
    
    logger.info("测试完成")
    return result

if __name__ == "__main__":
    logger.info("开始测试边界框归一化...")
    result = test_prepare_targets()
    if result is not None:
        logger.info("测试成功!")
    else:
        logger.error("测试失败!") 