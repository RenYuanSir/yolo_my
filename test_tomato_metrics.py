#!/usr/bin/env python3
"""
测试TomatoMetrics类的功能
"""

import torch
import numpy as np
from pathlib import Path
import sys
import os

# 添加项目路径
sys.path.append('.')

from ultralytics.utils.metrics import TomatoMetrics

def test_tomato_metrics():
    """测试TomatoMetrics的基本功能"""
    print("🧪 测试TomatoMetrics类...")
    
    # 创建TomatoMetrics实例
    metrics = TomatoMetrics(save_dir=Path("."), plot=False, names={0: "tomato"})
    
    # 模拟一些测试数据 - 修复维度问题
    # tp需要是二维张量，形状为(N, 10)，其中10是IoU阈值的数量
    # 使用浮点类型而不是布尔类型，因为ap_per_class函数中有减法操作
    tp = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    conf = torch.tensor([0.8, 0.9, 0.7, 0.6])
    pred_cls = torch.tensor([0, 0, 0, 0])
    target_cls = torch.tensor([0, 0, 0, 0])  # 确保长度匹配
    
    # 模拟h_pos数据
    pred_h = torch.tensor([0.3, 0.5, 0.2, 0.4])
    target_h = torch.tensor([0.35, 0.45, 0.25, 0.42])  # 确保长度匹配
    
    print(f"📊 测试数据:")
    print(f"    tp shape: {tp.shape}")
    print(f"    tp dtype: {tp.dtype}")
    print(f"    conf shape: {conf.shape}")
    print(f"    pred_cls shape: {pred_cls.shape}")
    print(f"    target_cls shape: {target_cls.shape}")
    print(f"    pred_h shape: {pred_h.shape}")
    print(f"    target_h shape: {target_h.shape}")
    
    # 测试process方法
    print("\n🔄 测试process方法...")
    try:
    metrics.process(tp, conf, pred_cls, target_cls, pred_h, target_h)
        print("    ✅ process方法执行成功")
    except Exception as e:
        print(f"    ❌ process方法执行失败: {e}")
    
    # 测试update_h_pos_stats方法
    print("\n📈 测试update_h_pos_stats方法...")
    try:
        metrics.update_h_pos_stats(tp[:2, 0].bool(), conf[:2], pred_h[:2], target_h[:2])
        print("    ✅ update_h_pos_stats方法执行成功")
    except Exception as e:
        print(f"    ❌ update_h_pos_stats方法执行失败: {e}")
    
    # 测试finalize_h_pos_metrics方法
    print("\n✅ 测试finalize_h_pos_metrics方法...")
    try:
    metrics.finalize_h_pos_metrics()
        print("    ✅ finalize_h_pos_metrics方法执行成功")
    except Exception as e:
        print(f"    ❌ finalize_h_pos_metrics方法执行失败: {e}")
    
    # 检查结果
    print(f"\n📋 结果:")
    print(f"    h_mae: {metrics.h_pos_stats.get('h_mae', 'N/A')}")
    print(f"    rank_acc: {metrics.h_pos_stats.get('rank_acc', 'N/A')}")
    print(f"    keys: {metrics.keys}")
    
    try:
        mean_results = metrics.mean_results()
        print(f"    mean_results: {mean_results}")
    except Exception as e:
        print(f"    ❌ mean_results执行失败: {e}")
    
    # 测试results_dict
    print(f"\n📊 results_dict:")
    try:
    results = metrics.results_dict
    for key, value in results.items():
        print(f"    {key}: {value}")
    except Exception as e:
        print(f"    ❌ results_dict执行失败: {e}")
    
    print("\n✅ TomatoMetrics测试完成！")

def test_ranking_accuracy():
    """测试排序准确率计算"""
    print("\n🧪 测试排序准确率计算...")
    
    metrics = TomatoMetrics()
    
    # 测试案例1: 完美排序
    pred_h = torch.tensor([0.1, 0.2, 0.3, 0.4])
    target_h = torch.tensor([0.1, 0.2, 0.3, 0.4])
    
    sorted_indices = torch.argsort(target_h)
    pred_sorted = pred_h[sorted_indices]
    correct_order = (pred_sorted[1:] >= pred_sorted[:-1]).sum()
    total_pairs = len(pred_sorted) - 1
    rank_acc = correct_order / total_pairs if total_pairs > 0 else 0.0
    
    print(f"    完美排序案例: {rank_acc:.4f} (期望: 1.0)")
    
    # 测试案例2: 完全错误排序
    pred_h = torch.tensor([0.4, 0.3, 0.2, 0.1])
    target_h = torch.tensor([0.1, 0.2, 0.3, 0.4])
    
    sorted_indices = torch.argsort(target_h)
    pred_sorted = pred_h[sorted_indices]
    correct_order = (pred_sorted[1:] >= pred_sorted[:-1]).sum()
    total_pairs = len(pred_sorted) - 1
    rank_acc = correct_order / total_pairs if total_pairs > 0 else 0.0
    
    print(f"    完全错误排序案例: {rank_acc:.4f} (期望: 0.0)")
    
    # 测试案例3: 部分正确排序
    pred_h = torch.tensor([0.2, 0.1, 0.4, 0.3])
    target_h = torch.tensor([0.1, 0.2, 0.3, 0.4])
    
    sorted_indices = torch.argsort(target_h)
    pred_sorted = pred_h[sorted_indices]
    correct_order = (pred_sorted[1:] >= pred_sorted[:-1]).sum()
    total_pairs = len(pred_sorted) - 1
    rank_acc = correct_order / total_pairs if total_pairs > 0 else 0.0
    
    print(f"    部分正确排序案例: {rank_acc:.4f}")

def test_h_pos_mae():
    """测试H-MAE计算"""
    print("\n🧪 测试H-MAE计算...")
    
    # 测试案例1: 完美预测
    pred_h = torch.tensor([0.1, 0.2, 0.3, 0.4])
    target_h = torch.tensor([0.1, 0.2, 0.3, 0.4])
    h_mae = torch.mean(torch.abs(pred_h - target_h)).item()
    print(f"    完美预测案例: {h_mae:.4f} (期望: 0.0)")
    
    # 测试案例2: 有误差的预测
    pred_h = torch.tensor([0.15, 0.25, 0.35, 0.45])
    target_h = torch.tensor([0.1, 0.2, 0.3, 0.4])
    h_mae = torch.mean(torch.abs(pred_h - target_h)).item()
    print(f"    有误差预测案例: {h_mae:.4f} (期望: 0.05)")

if __name__ == "__main__":
    test_tomato_metrics()
    test_ranking_accuracy()
    test_h_pos_mae()
    print("\n🎉 所有测试完成！") 