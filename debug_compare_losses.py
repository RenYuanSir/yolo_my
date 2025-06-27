#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
比较修复前后的损失计算效果
"""

import os
import sys
import torch
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')

# 设置日志
log_dir = Path('debug_logs')
log_dir.mkdir(exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f'loss_comparison_{timestamp}.log'

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('loss_comparison')

# 添加YOLO路径
YOLO_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, YOLO_PATH)

try:
    # 导入必要的模块
    from ultralytics.nn.tasks import TomatoDetectionModel
    from ultralytics.utils.loss import TomatoDetectWithRankLoss
    logger.info("成功导入YOLO模块")
except ImportError as e:
    logger.error(f"导入YOLO模块失败: {e}")
    sys.exit(1)

def load_model():
    """加载模型和创建损失函数"""
    model_yaml = os.path.join(YOLO_PATH, "ultralytics/cfg/models/v12/yolo12-A2C2f-DYT-EfficientHead-MambaOut.yaml")
    
    # 加载模型
    logger.info(f"加载模型: {model_yaml}")
    model = TomatoDetectionModel(cfg=model_yaml, nc=6)
    logger.info("模型加载成功")
    
    # 为模型添加缺少的args属性
    class Args:
        def __init__(self):
            self.box = 7.5  # 边界框损失权重
            self.cls = 0.5  # 分类损失权重
            self.dfl = 1.5  # DFL损失权重
            self.device = next(model.parameters()).device
    
    # 添加args属性
    model.args = Args()
    
    # 修复模型数据类型问题
    for buffer in model.buffers():
        if buffer.dtype != torch.float32:
            buffer.data = buffer.data.to(torch.float32)
    
    # 创建损失计算器（原始版本）
    loss_fn_original = TomatoDetectWithRankLoss(model)
    logger.info("原始损失函数已创建")
    
    return model, loss_fn_original

def create_sample_batch(batch_size=2, num_objects=5, num_clusters=2):
    """创建一个样本批次，用于测试损失计算"""
    # 创建图像
    img = torch.rand(batch_size, 3, 640, 640)
    
    # 创建边界框和标签
    total_objects = batch_size * num_objects
    batch_idx = torch.cat([torch.ones(num_objects) * i for i in range(batch_size)])
    
    # 随机生成类别，范围0-5
    cls = torch.randint(0, 6, (total_objects, 1)).float()
    
    # 随机生成边界框，格式为xywh，范围0-1
    bboxes = torch.rand(total_objects, 4)
    # 确保宽高不会太小
    bboxes[:, 2:] = torch.clamp(bboxes[:, 2:], min=0.1, max=0.5)
    
    # 创建cluster_ids：每个批次有几个串
    cluster_ids = torch.zeros(total_objects, 1)
    objects_per_cluster = num_objects // num_clusters
    for i in range(batch_size):
        for j in range(num_clusters):
            start_idx = i * num_objects + j * objects_per_cluster
            end_idx = start_idx + objects_per_cluster
            cluster_ids[start_idx:end_idx] = j + 1
    
    # 创建h_rel：在每个串内，根据类别值设置高度（成熟度越高的位置越高）
    h_rel = torch.zeros(total_objects, 1)
    for i in range(batch_size):
        for j in range(num_clusters):
            start_idx = i * num_objects + j * objects_per_cluster
            end_idx = start_idx + objects_per_cluster
            
            # 根据类别值归一化高度（类别值越大，高度越高）
            cluster_cls = cls[start_idx:end_idx]
            min_cls = cluster_cls.min()
            max_cls = cluster_cls.max()
            if max_cls > min_cls:
                norm_cls = (cluster_cls - min_cls) / (max_cls - min_cls)
                # 将归一化的类别映射到高度范围0.2-0.8
                h_rel[start_idx:end_idx] = 0.2 + 0.6 * norm_cls
            else:
                # 如果类别都相同，均匀分布高度
                for k in range(objects_per_cluster):
                    h_rel[start_idx + k] = 0.2 + 0.6 * (k / (objects_per_cluster - 1)) if objects_per_cluster > 1 else 0.5
    
    # 组合为batch
    batch = {
        'img': img,
        'batch_idx': batch_idx,
        'cls': cls,
        'bboxes': bboxes,
        'cluster_ids': cluster_ids,
        'h_rel': h_rel,
    }
    
    return batch

def apply_patch():
    """应用修复补丁"""
    try:
        # 导入补丁模块
        from tomato_model_fix_patch import apply_patches
        
        # 应用补丁
        success = apply_patches()
        logger.info(f"补丁应用{'成功' if success else '失败'}")
        
        return success
    except Exception as e:
        logger.error(f"应用补丁失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def calculate_loss(model, loss_fn, batch, is_patched=False):
    """计算损失"""
    try:
        # 在评估模式下执行前向传播
        model.eval()
        with torch.no_grad():
            preds = model(batch['img'])
        
        # 计算损失
        loss, loss_items = loss_fn(preds, batch)
        
        # 提取损失项 - 确保正确处理多维张量
        if loss_items.dim() > 1:  # 如果是多维张量
            result = {
                'total_loss': loss.item(),
                'box_loss': loss_items[0, 0].item(),
                'cls_loss': loss_items[0, 1].item(),
                'dfl_loss': loss_items[0, 2].item(),
                'rank_loss': loss_items[0, 3].item(),
            }
        else:  # 如果是一维张量
            result = {
                'total_loss': loss.item(),
                'box_loss': loss_items[0].item(),
                'cls_loss': loss_items[1].item(),
                'dfl_loss': loss_items[2].item(),
                'rank_loss': loss_items[3].item(),
            }
        
        # 判断损失是否有效
        is_valid = True
        for name, value in result.items():
            if torch.isnan(torch.tensor(value)) or torch.isinf(torch.tensor(value)):
                logger.warning(f"{'修复后' if is_patched else '原始'} {name} 无效: {value}")
                is_valid = False
                
        result['valid'] = is_valid
        return result
    
    except Exception as e:
        logger.error(f"{'修复后' if is_patched else '原始'}损失计算失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'total_loss': float('nan'),
            'box_loss': float('nan'),
            'cls_loss': float('nan'),
            'dfl_loss': float('nan'),
            'rank_loss': float('nan'),
            'valid': False
        }

def compare_losses(model, original_loss_fn, patched_loss_fn, batch):
    """比较原始和修复后的损失"""
    # 计算原始损失
    logger.info("计算原始损失...")
    original_result = calculate_loss(model, original_loss_fn, batch)
    
    # 计算修复后的损失
    logger.info("计算修复后的损失...")
    patched_result = calculate_loss(model, patched_loss_fn, batch, is_patched=True)
    
    # 打印结果
    logger.info("\n" + "=" * 80)
    logger.info("损失比较:")
    logger.info("-" * 40)
    
    logger.info(f"{'项目':<15}{'原始损失':<15}{'修复后损失':<15}{'是否改进':<10}")
    logger.info("-" * 60)
    
    for name in ['total_loss', 'box_loss', 'cls_loss', 'dfl_loss', 'rank_loss']:
        original_value = original_result[name]
        patched_value = patched_result[name]
        
        # 判断是否改进
        if name in ['box_loss', 'dfl_loss']:
            # 对于这些损失，从0变为正值是改进
            is_improved = patched_value > 0 and original_value == 0
        elif name == 'rank_loss':
            # 对于排序损失，不再固定为0.01是改进
            is_improved = abs(patched_value - 0.01) > 1e-6
        elif name == 'cls_loss':
            # 对于分类损失，不再固定为1是改进
            is_improved = abs(patched_value - 1.0) > 1e-6
        else:
            # 对于总损失，有效值是改进
            is_improved = not torch.isnan(torch.tensor(patched_value)) and not torch.isinf(torch.tensor(patched_value))
            
        status = "✓" if is_improved else "✗" if patched_value == original_value else "?"
            
        logger.info(f"{name:<15}{original_value:<15.4f}{patched_value:<15.4f}{status:<10}")
    
    logger.info("=" * 80)
    
    return original_result, patched_result

def visualize_comparison(original_result, patched_result, save_path='debug_logs/loss_comparison.png'):
    """可视化损失比较"""
    # 检查结果是否有效
    if not original_result['valid'] and not patched_result['valid']:
        logger.warning("两种损失都无效，无法生成可视化")
        return
    
    # 准备数据
    loss_names = ['box_loss', 'cls_loss', 'dfl_loss', 'rank_loss', 'total_loss']
    original_values = [original_result[name] for name in loss_names]
    patched_values = [patched_result[name] for name in loss_names]
    
    # 替换无效值
    for i, v in enumerate(original_values):
        if torch.isnan(torch.tensor(v)) or torch.isinf(torch.tensor(v)):
            original_values[i] = 0
    
    for i, v in enumerate(patched_values):
        if torch.isnan(torch.tensor(v)) or torch.isinf(torch.tensor(v)):
            patched_values[i] = 0
    
    # 设置图表
    plt.figure(figsize=(10, 6))
    
    # 在同一图中绘制条形图
    x = np.arange(len(loss_names))
    width = 0.35
    
    plt.bar(x - width/2, original_values, width, label='原始损失')
    plt.bar(x + width/2, patched_values, width, label='修复后损失')
    
    plt.xlabel('损失类型')
    plt.ylabel('损失值')
    plt.title('损失比较：原始 vs 修复后')
    plt.xticks(x, loss_names, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"损失比较图已保存至: {save_path}")

def main():
    """主函数"""
    logger.info("开始比较损失计算效果...")
    
    # 加载模型和原始损失函数
    model, original_loss_fn = load_model()
    
    # 创建样本批次
    batch = create_sample_batch(batch_size=2, num_objects=10, num_clusters=3)
    logger.info(f"已创建样本批次: batch_size={len(batch['img'])}, num_objects={len(batch['batch_idx'])}, num_clusters={len(torch.unique(batch['cluster_ids']))}")
    
    # 保存原始损失函数的引用
    import copy
    original_methods = {
        'compute_ranking_loss': original_loss_fn.compute_ranking_loss,
        '__call__': original_loss_fn.__call__
    }
    
    # 应用补丁
    success = apply_patch()
    
    if success:
        # 创建补丁后的损失函数（补丁已全局应用）
        patched_loss_fn = TomatoDetectWithRankLoss(model)
        
        # 比较损失
        original_result, patched_result = compare_losses(model, original_loss_fn, patched_loss_fn, batch)
        
        # 可视化比较
        visualize_comparison(original_result, patched_result)
        
        logger.info("损失比较完成")
    else:
        logger.error("应用补丁失败，无法比较损失")

if __name__ == "__main__":
    main() 