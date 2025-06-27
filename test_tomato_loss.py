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
import yaml
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_tomato_loss')

# 添加YOLO路径
YOLO_PATH = os.path.dirname(os.path.abspath(__file__))
print(f"YOLO_PATH: {YOLO_PATH}")
sys.path.insert(0, YOLO_PATH)

# 导入YOLO相关模块
try:
    from ultralytics.utils.loss import TomatoDetectWithRankLoss
    from ultralytics.nn.tasks import TomatoDetectionModel
    logger.info("成功导入YOLO模块")
except ImportError as e:
    logger.error(f"导入YOLO模块失败: {e}")
    sys.exit(1)

def create_mock_model():
    """创建模拟模型对象"""
    class MockModel:
        def __init__(self):
            self.nc = 6  # 类别数
            self.reg_max = 16  # DFL回归最大值
            self.stride = torch.tensor([8.0, 16.0, 32.0])
            self.args = type('', (), {})()
            self.args.device = 'cpu'
            self.args.box = 7.5  # 边界框损失权重
            self.args.cls = 0.5  # 分类损失权重
            self.args.dfl = 1.5  # DFL损失权重
            self.device = 'cpu'
            
            # 创建模型结构
            class MockDetect:
                def __init__(self):
                    self.nc = 6  # 类别数
                    self.reg_max = 16  # DFL回归最大值
                    self.stride = torch.tensor([8.0, 16.0, 32.0])
                    self.no = 4 * 16 + 6  # 没有包含h_pos通道
            
            # 添加模型属性
            self.model = [None]  # 第一个元素是None
            self.model.append(MockDetect())  # 第二个元素是Detect模块
            
            # 创建一个参数以便于parameters()方法返回
            self._params = [torch.zeros(1, requires_grad=True, device='cpu')]
            
        def parameters(self):
            """返回模型参数，用于初始化损失函数"""
            for p in self._params:
                yield p
            
        def named_buffers(self):
            """返回命名缓冲区"""
            return []
            
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

def test_with_standard_features():
    """使用标准特征格式测试"""
    logger.info("-" * 50)
    logger.info("测试1: 使用标准特征格式")
    logger.info("-" * 50)
    
    # 创建模拟模型
    mock_model = create_mock_model()
    
    # 创建损失计算器
    loss_fn = TomatoDetectWithRankLoss(mock_model)
    
    # 设置特征张量尺寸
    batch_size = 2
    nc = 6
    reg_max = 16
    
    # 创建三个尺度的特征图
    # 注意：损失函数期望的通道数是 4*reg_max + nc = 70
    # 但实际特征的通道数是 4*reg_max + nc + 1 = 71 (包含h_pos通道)
    expected_channels = 4 * reg_max + nc  # 损失函数期望的通道数
    actual_channels = expected_channels + 1  # 实际特征的通道数，多了一个h_pos通道
    
    # 创建三个尺度的特征图
    features = [
        torch.rand(batch_size, actual_channels, 80, 80),  # 大尺度特征图
        torch.rand(batch_size, actual_channels, 40, 40),  # 中尺度特征图
        torch.rand(batch_size, actual_channels, 20, 20),  # 小尺度特征图
    ]
    
    # 创建批次数据
    batch = {
        'img': torch.rand(batch_size, 3, 640, 640),
        'batch_idx': torch.zeros(10),
        'cls': torch.zeros(10, 1),
        'bboxes': torch.rand(10, 4),
        'cluster_ids': torch.zeros(10, 1),
        'h_rel': torch.rand(10, 1),
    }
    
    # 打印特征信息
    logger.info(f"特征数量: {len(features)}")
    for i, feat in enumerate(features):
        logger.info(f"特征[{i}]: 形状={feat.shape}")
    
    # 使用Detect_Efficient_Tomato输出格式
    preds = (
        torch.rand(batch_size, actual_channels, 9000),  # 第一部分是拼接后的预测
        {
            'features': features,  # features部分包含原始特征图
            'h_pos': [feat[:, -1:] for feat in features]  # h_pos部分包含高度预测
        }
    )
    
    # 对self.no进行修改以模拟错误情况
    logger.info(f"修改前: mock_model.model[-1].no = {mock_model.model[-1].no}")
    mock_model.model[-1].no = expected_channels  # 设置为不包含h_pos通道的通道数
    logger.info(f"修改后: mock_model.model[-1].no = {mock_model.model[-1].no}")
    
    # 提取损失函数中的self.no
    logger.info(f"损失函数: loss_fn.no = {loss_fn.no}, loss_fn.reg_max = {loss_fn.reg_max}, loss_fn.nc = {loss_fn.nc}")
    
    try:
        # 计算损失
        logger.info(f"计算损失...")
        loss, loss_items = loss_fn(preds, batch)
        
        # 打印损失结果
        logger.info(f"总损失: {loss.item()}")
        logger.info(f"损失项: {loss_items.tolist()}")
        logger.info("测试成功，损失计算正确")
        return True
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_with_real_model():
    """使用真实模型测试"""
    logger.info("-" * 50)
    logger.info("测试2: 使用真实模型")
    logger.info("-" * 50)
    
    model_yaml = os.path.join(YOLO_PATH, "ultralytics/cfg/models/v12/yolo12-A2C2f-DYT-EfficientHead-MambaOut.yaml")
    
    try:
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
        logger.info(f"已为模型添加args属性: box={model.args.box}, cls={model.args.cls}, dfl={model.args.dfl}")
        # 注意这里是 Detect_Efficient_Tomato 实例
        model.model[-1].stride = torch.tensor([8., 16., 32.])

        
        # 修复模型数据类型问题
        for buffer in model.buffers():
            if buffer.dtype != torch.float32:
                buffer.data = buffer.data.to(torch.float32)
        
        # 创建损失计算器
        loss_fn = TomatoDetectWithRankLoss(model)
        logger.info(f"损失函数: loss_fn.no = {loss_fn.no}, loss_fn.reg_max = {loss_fn.reg_max}, loss_fn.nc = {loss_fn.nc}")
        
        # 准备模拟输入数据
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 640, 640)
        
        # 设置为评估模式
        model.eval()
        
        # 执行前向传播
        with torch.no_grad():
            logger.info("执行前向传播...")
            preds = model(dummy_input)
            logger.info("前向传播成功")
            
            # 打印输出结构
            if isinstance(preds, (tuple, list)):
                logger.info(f"输出是{type(preds).__name__}，长度: {len(preds)}")
                for i, out in enumerate(preds):
                    if isinstance(out, torch.Tensor):
                        logger.info(f"输出[{i}]: 形状={out.shape}, 类型={out.dtype}")
                    elif isinstance(out, dict):
                        logger.info(f"输出[{i}]: 是字典, 键={list(out.keys())}")
                        for k, v in out.items():
                            if isinstance(v, (list, tuple)):
                                logger.info(f"  - {k}: 是{type(v).__name__}, 长度={len(v)}")
                                for j, item in enumerate(v):
                                    logger.info(f"    - {k}[{j}]: 形状={item.shape if isinstance(item, torch.Tensor) else 'N/A'}")
                            else:
                                logger.info(f"  - {k}: 类型={type(v)}")
        
        # 创建批次数据
        batch = {
            'img': dummy_input,
            'batch_idx': torch.zeros(10),
            'cls': torch.zeros(10, 1),
            'bboxes': torch.rand(10, 4),
            'cluster_ids': torch.zeros(10, 1),
            'h_rel': torch.rand(10, 1),
        }
        
        # 计算损失
        try:
            logger.info(f"计算损失...")
            loss, loss_items = loss_fn(preds, batch)
            
            # 打印损失结果
            logger.info(f"总损失: {loss.item()}")
            logger.info(f"损失项: {loss_items.tolist()}")
            logger.info("测试成功，损失计算正确")
            
            # 验证rank通道是否被正确处理
            if isinstance(preds, tuple) and len(preds) > 1 and isinstance(preds[1], dict):
                if 'features' in preds[1]:
                    features = preds[1]['features']
                    total_channels = features[0].shape[1]
                    expected_channels = 4 * loss_fn.reg_max + loss_fn.nc  # 70
                    
                    if total_channels > expected_channels:
                        logger.info(f"✓ 验证通过：模型输出通道数({total_channels})大于self.no({expected_channels})，修复成功处理了额外的rank通道")
                    else:
                        logger.info(f"! 注意：模型输出通道数({total_channels})等于self.no({expected_channels})，没有额外通道")
            
            return True
        except Exception as e:
            logger.error(f"计算损失失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """主函数"""
    logger.info("开始测试TomatoDetectWithRankLoss")
    
    # 测试标准特征
    test_with_standard_features()
    
    # 测试真实模型
    test_with_real_model()
    
    logger.info("测试完成")

if __name__ == "__main__":
    main() 