#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试修复后的排序损失函数
"""

import os
import sys
import torch
import logging
from pathlib import Path
import traceback

# 确保目录存在
log_dir = Path('debug_logs')
log_dir.mkdir(exist_ok=True)
log_file = log_dir / 'ranking_loss_test.log'

# 清空已有日志文件
with open(log_file, 'w') as f:
    f.write("开始测试排序损失函数\n")

# 设置日志，确保立即刷新
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('rank_loss_test')

# 测试日志系统
logger.info("日志系统初始化成功")
logger.info("=" * 80)
logger.info("测试修复后的排序损失函数")
logger.info("=" * 80)
logger.info(f"工作目录: {os.getcwd()}")
logger.info(f"YOLO路径: {os.path.dirname(os.path.abspath(__file__))}")
logger.info(f"Python版本: {sys.version}")

# 添加YOLO路径
YOLO_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, YOLO_PATH)
logger.info(f"已添加YOLO路径到sys.path: {YOLO_PATH}")

# 导入修复模块
try:
    logger.info("正在导入YOLO模块和修复函数...")
    from loss_fix import compute_ranking_loss_fixed, apply_computation_patches
    from ultralytics.utils.loss import TomatoDetectWithRankLoss
    from ultralytics.nn.tasks import TomatoDetectionModel
    logger.info("成功导入YOLO模块和修复函数")
except ImportError as e:
    logger.error(f"导入模块失败: {e}")
    sys.exit(1)

def load_model():
    """加载模型并准备相关组件"""
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
    for name, buffer in model.named_buffers():
        if buffer.dtype != torch.float32:
            logger.info(f"转换buffer {name} 从 {buffer.dtype} 到 float32")
            buffer.data = buffer.data.to(torch.float32)
    
    # 创建损失计算器
    loss_fn = TomatoDetectWithRankLoss(model)
    logger.info(f"损失函数: loss_fn.no = {loss_fn.no}, loss_fn.reg_max = {loss_fn.reg_max}, loss_fn.nc = {loss_fn.nc}")
    
    return model, loss_fn

def create_sample_batch(case_id=1):
    """创建测试批次，根据case_id创建不同的测试场景"""
    if case_id == 1:
        # 场景1: 单个串内的番茄，高度与成熟度呈正相关（符合期望）
        batch = {
            'img': torch.rand(1, 3, 640, 640),
            'batch_idx': torch.zeros(3),
            'cls': torch.tensor([[0], [1], [2]]),  # 成熟度依次增加
            'bboxes': torch.tensor([
                [0.5, 0.2, 0.1, 0.1],  # 顶部番茄
                [0.5, 0.5, 0.1, 0.1],  # 中间番茄
                [0.5, 0.8, 0.1, 0.1],  # 底部番茄
            ]),
            'cluster_ids': torch.ones(3, 1),  # 同一串
            'h_rel': torch.tensor([[0.8], [0.5], [0.2]]),  # 高度从高到低
        }
        logger.info("场景1: 高度与成熟度呈正相关（符合期望）")
        
    elif case_id == 2:
        # 场景2: 单个串内的番茄，高度与成熟度呈负相关（违反期望）
        batch = {
            'img': torch.rand(1, 3, 640, 640),
            'batch_idx': torch.zeros(3),
            'cls': torch.tensor([[2], [1], [0]]),  # 成熟度依次降低
            'bboxes': torch.tensor([
                [0.5, 0.2, 0.1, 0.1],  # 顶部番茄
                [0.5, 0.5, 0.1, 0.1],  # 中间番茄
                [0.5, 0.8, 0.1, 0.1],  # 底部番茄
            ]),
            'cluster_ids': torch.ones(3, 1),  # 同一串
            'h_rel': torch.tensor([[0.8], [0.5], [0.2]]),  # 高度从高到低
        }
        logger.info("场景2: 高度与成熟度呈负相关（违反期望）")
        
    elif case_id == 3:
        # 场景3: 多个串，每个串内番茄排序和成熟度不一致
        batch = {
            'img': torch.rand(1, 3, 640, 640),
            'batch_idx': torch.zeros(6),
            'cls': torch.tensor([[0], [1], [2], [2], [1], [0]]),
            'bboxes': torch.tensor([
                [0.3, 0.2, 0.1, 0.1],  # 串1顶部
                [0.3, 0.5, 0.1, 0.1],  # 串1中间
                [0.3, 0.8, 0.1, 0.1],  # 串1底部
                [0.7, 0.2, 0.1, 0.1],  # 串2顶部
                [0.7, 0.5, 0.1, 0.1],  # 串2中间
                [0.7, 0.8, 0.1, 0.1],  # 串2底部
            ]),
            'cluster_ids': torch.tensor([[1], [1], [1], [2], [2], [2]]),
            'h_rel': torch.tensor([[0.8], [0.5], [0.2], [0.8], [0.5], [0.2]]),
        }
        logger.info("场景3: 多个串，每个串内排序不同")
        
    elif case_id == 4:
        # 场景4: 边界情况 - 所有番茄都是同一类别
        batch = {
            'img': torch.rand(1, 3, 640, 640),
            'batch_idx': torch.zeros(3),
            'cls': torch.ones(3, 1),  # 所有都是类别1
            'bboxes': torch.tensor([
                [0.5, 0.2, 0.1, 0.1],
                [0.5, 0.5, 0.1, 0.1],
                [0.5, 0.8, 0.1, 0.1],
            ]),
            'cluster_ids': torch.ones(3, 1),
            'h_rel': torch.tensor([[0.8], [0.5], [0.2]]),
        }
        logger.info("场景4: 边界情况 - 所有番茄都是同一类别")
        
    elif case_id == 5:
        # 场景5: 边界情况 - 串ID为0（通常视为背景）
        batch = {
            'img': torch.rand(1, 3, 640, 640),
            'batch_idx': torch.zeros(3),
            'cls': torch.tensor([[0], [1], [2]]),
            'bboxes': torch.tensor([
                [0.5, 0.2, 0.1, 0.1],
                [0.5, 0.5, 0.1, 0.1],
                [0.5, 0.8, 0.1, 0.1],
            ]),
            'cluster_ids': torch.zeros(3, 1),
            'h_rel': torch.tensor([[0.8], [0.5], [0.2]]),
        }
        logger.info("场景5: 边界情况 - 串ID为0（通常视为背景）")
        
    else:
        # 默认场景
        batch = {
            'img': torch.rand(1, 3, 640, 640),
            'batch_idx': torch.zeros(1),
            'cls': torch.zeros(1, 1),
            'bboxes': torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
            'cluster_ids': torch.ones(1, 1),
            'h_rel': torch.tensor([[0.5]]),
        }
        logger.info("默认场景: 单个番茄")
    
    return batch

def create_simulated_h_pos_features(case_id=1):
    """创建模拟的h_pos特征，以测试基于特征的排序损失"""
    if case_id == 1:
        # 符合排序预期的h_pos特征（从上到下递增）
        h_pos_list = [
            torch.linspace(0, 1, 10).reshape(1, 1, 10, 1).repeat(1, 1, 1, 10),
            torch.linspace(0, 1, 5).reshape(1, 1, 5, 1).repeat(1, 1, 1, 5),
        ]
        logger.info("h_pos特征: 符合预期排序（从上到下递增）")
        
    elif case_id == 2:
        # 违反排序预期的h_pos特征（从上到下递减）
        h_pos_list = [
            torch.linspace(1, 0, 10).reshape(1, 1, 10, 1).repeat(1, 1, 1, 10),
            torch.linspace(1, 0, 5).reshape(1, 1, 5, 1).repeat(1, 1, 1, 5),
        ]
        logger.info("h_pos特征: 违反预期排序（从上到下递减）")
        
    elif case_id == 3:
        # 部分违反排序预期的h_pos特征
        h_pos1 = torch.zeros(1, 1, 10, 10)
        h_pos1[0, 0, :5, :] = torch.linspace(0, 0.5, 5).reshape(5, 1).repeat(1, 10)
        h_pos1[0, 0, 5:, :] = torch.linspace(0.4, 0.9, 5).reshape(5, 1).repeat(1, 10)
        
        h_pos2 = torch.zeros(1, 1, 5, 5)
        h_pos2[0, 0, :3, :] = torch.linspace(0, 0.5, 3).reshape(3, 1).repeat(1, 5)
        h_pos2[0, 0, 3:, :] = torch.linspace(0.3, 0.6, 2).reshape(2, 1).repeat(1, 5)
        
        h_pos_list = [h_pos1, h_pos2]
        logger.info("h_pos特征: 部分违反预期排序")
        
    else:
        # 默认：随机h_pos特征
        h_pos_list = [
            torch.rand(1, 1, 10, 10),
            torch.rand(1, 1, 5, 5),
        ]
        logger.info("h_pos特征: 随机值")
    
    return h_pos_list

def test_original_ranking_loss():
    """测试原始排序损失函数"""
    logger.info("=" * 80)
    logger.info("测试原始排序损失函数")
    logger.info("=" * 80)
    
    _, loss_fn = load_model()
    
    for i in range(1, 6):
        batch = create_sample_batch(i)
        loss = loss_fn.compute_ranking_loss(batch)
        logger.info(f"原始函数 - 场景{i}排序损失: {loss.item()}")

def test_fixed_ranking_loss():
    """测试修复后的排序损失函数"""
    logger.info("=" * 80)
    logger.info("测试修复后的排序损失函数")
    logger.info("=" * 80)
    
    _, loss_fn = load_model()
    
    # 临时绑定修复函数到实例
    old_method = loss_fn.compute_ranking_loss
    loss_fn.compute_ranking_loss = lambda batch, pred_scores=None, h_pos_features=None: compute_ranking_loss_fixed(loss_fn, batch, pred_scores, h_pos_features)
    
    try:
        for i in range(1, 6):
            batch = create_sample_batch(i)
            loss = loss_fn.compute_ranking_loss(batch)
            logger.info(f"修复函数 - 场景{i}排序损失: {loss.item()}")
        
        # 测试基于h_pos特征的排序损失
        for i in range(1, 4):
            h_pos_features = create_simulated_h_pos_features(i)
            batch = create_sample_batch(1)  # 这里使用任意batch
            loss = loss_fn.compute_ranking_loss(batch, h_pos_features=h_pos_features)
            logger.info(f"修复函数 - h_pos特征场景{i}排序损失: {loss.item()}")
    finally:
        # 恢复原始方法
        loss_fn.compute_ranking_loss = old_method

def test_global_patch():
    """测试全局补丁应用"""
    logger.info("=" * 80)
    logger.info("测试全局补丁应用")
    logger.info("=" * 80)
    
    # 应用全局补丁
    success = apply_computation_patches()
    logger.info(f"补丁应用{'成功' if success else '失败'}")
    
    if success:
        _, loss_fn = load_model()
        
        # 测试应用补丁后的函数
        for i in range(1, 4):
            batch = create_sample_batch(i)
            loss = loss_fn.compute_ranking_loss(batch)
            logger.info(f"全局补丁 - 场景{i}排序损失: {loss.item()}")

def main():
    """主函数"""
    logger.info("开始测试排序损失函数")
    
    # 测试原始排序损失函数
    test_original_ranking_loss()
    
    # 测试修复后的排序损失函数
    test_fixed_ranking_loss()
    
    # 测试全局补丁应用
    test_global_patch()
    
    logger.info("测试完成")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        logger.error(traceback.format_exc()) 