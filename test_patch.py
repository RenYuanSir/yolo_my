#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试修复补丁效果
"""

import os
import sys
import torch
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("test_patch")

# 添加YOLO路径
YOLO_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, YOLO_PATH)

def create_sample_batch(batch_size=1):
    """创建一个样本批次用于测试"""
    # 创建图像
    img = torch.rand(batch_size, 3, 640, 640)
    
    # 创建标签
    batch = {
        'img': img,
        'batch_idx': torch.zeros(3),
        'cls': torch.tensor([[0], [1], [2]]),  # 成熟度依次变低
        'bboxes': torch.tensor([
            [0.5, 0.2, 0.1, 0.1],  # 顶部番茄
            [0.5, 0.5, 0.1, 0.1],  # 中间番茄
            [0.5, 0.8, 0.1, 0.1],  # 底部番茄
        ]),
        'cluster_ids': torch.ones(3, 1),  # 同一串
        'h_rel': torch.tensor([[0.2], [0.5], [0.8]]),  # 高度从上到下递增
    }
    
    return batch

def main():
    """主函数"""
    try:
        logger.info("测试补丁应用...")
        
        # 导入补丁
        from tomato_model_fix_patch import apply_patches, compute_ranking_loss_fixed
        
        # 应用补丁
        success = apply_patches()
        logger.info(f"应用补丁: {'成功' if success else '失败'}")
        
        # 测试排序损失计算
        mock_self = type('', (), {'device': 'cpu', 'margin': 0.1})()
        batch = create_sample_batch()
        
        # 计算排序损失
        rank_loss = compute_ranking_loss_fixed(mock_self, batch)
        logger.info(f"排序损失计算: loss = {rank_loss.item()}")
        
        # 验证对正确排序的情况下损失应该接近最小值
        is_correct = rank_loss.item() <= 0.02
        logger.info(f"损失验证: {'通过 ✓' if is_correct else '失败 ✗'}")
        
        # 修改batch中的类别顺序，让排序违反预期（上方番茄类别值大于下方番茄）
        batch['cls'] = torch.tensor([[2], [1], [0]])  # 修改为成熟度从上到下递增（与预期相反）
        
        # 计算排序损失
        rank_loss_bad = compute_ranking_loss_fixed(mock_self, batch)
        logger.info(f"错误排序损失计算: loss = {rank_loss_bad.item()}")
        
        # 验证不正确排序的损失应该较大
        is_higher = rank_loss_bad.item() > rank_loss.item()
        logger.info(f"损失对比验证: {'通过 ✓' if is_higher else '失败 ✗'}")
        
        # 测试h_pos特征排序损失计算
        # 创建模拟的h_pos特征（从上到下递增）
        h_pos_good = [
            torch.linspace(0, 1, 10).reshape(1, 1, 10, 1).repeat(1, 1, 1, 10)
        ]
        
        rank_loss_h_pos_good = compute_ranking_loss_fixed(mock_self, batch, h_pos_features=h_pos_good)
        logger.info(f"h_pos正确排序损失计算: loss = {rank_loss_h_pos_good.item()}")
        
        # 创建模拟的h_pos特征（从上到下递减，违反预期）
        h_pos_bad = [
            torch.linspace(1, 0, 10).reshape(1, 1, 10, 1).repeat(1, 1, 1, 10)
        ]
        
        rank_loss_h_pos_bad = compute_ranking_loss_fixed(mock_self, batch, h_pos_features=h_pos_bad)
        logger.info(f"h_pos错误排序损失计算: loss = {rank_loss_h_pos_bad.item()}")
        
        # 验证不正确h_pos排序的损失应该较大
        is_h_pos_higher = rank_loss_h_pos_bad.item() > rank_loss_h_pos_good.item()
        logger.info(f"h_pos损失对比验证: {'通过 ✓' if is_h_pos_higher else '失败 ✗'}")
        
        # 总体结果
        all_passed = is_correct and is_higher and is_h_pos_higher
        logger.info(f"测试总结: {'全部通过 ✓' if all_passed else '部分失败 ✗'}")
        
        logger.info("补丁测试完成")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 