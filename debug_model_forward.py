#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import yaml
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('debug_model_forward')

# 添加YOLO路径
YOLO_PATH = os.path.dirname(os.path.abspath(__file__))
print(f"YOLO_PATH: {YOLO_PATH}")
sys.path.insert(0, YOLO_PATH)  # 确保本地路径优先

# 导入YOLO相关模块
try:
    from ultralytics.nn.tasks import TomatoDetectionModel
    from ultralytics.utils.loss import TomatoDetectWithRankLoss
    from ultralytics.utils.torch_utils import intersect_dicts
    logger.info("成功导入YOLO模块")
except ImportError as e:
    logger.error(f"导入YOLO模块失败: {e}")
    sys.exit(1)

def print_tensor_info(name, tensor):
    """打印张量的详细信息"""
    if isinstance(tensor, torch.Tensor):
        info = (
            f"{name}:\n"
            f"- 形状: {tensor.shape}\n"
            f"- 数据类型: {tensor.dtype}\n"
            f"- 设备: {tensor.device}\n"
            f"- 值范围: [{tensor.min().item():.6f}, {tensor.max().item():.6f}]"
        )
        logger.info(info)
    elif isinstance(tensor, (list, tuple)):
        logger.info(f"{name}: 是列表/元组，长度={len(tensor)}")
        for i, item in enumerate(tensor):
            print_tensor_info(f"{name}[{i}]", item)
    elif isinstance(tensor, dict):
        logger.info(f"{name}: 是字典，键={list(tensor.keys())}")
        for k, v in tensor.items():
            print_tensor_info(f"{name}['{k}']", v)
    else:
        logger.info(f"{name}: 类型={type(tensor)}")

def fix_model_dtype_issue(model):
    """修复模型中的数据类型问题"""
    logger.info("尝试修复模型数据类型问题...")
    
    # 检查所有参数
    for name, param in model.named_parameters():
        if param.dtype == torch.float32:
            logger.info(f"参数 {name} 的类型: {param.dtype}")
        else:
            logger.warning(f"参数 {name} 的类型: {param.dtype} (非float32)")
    
    # 强制转换所有buffer为float32
    for name, buffer in model.named_buffers():
        if buffer.dtype != torch.float32:
            logger.warning(f"转换buffer {name} 从 {buffer.dtype} 到 float32")
            buffer.data = buffer.data.to(torch.float32)
    
    # 特别检查anchors相关buffer，这些通常需要是整数
    for name, buffer in model.named_buffers():
        if "anchor" in name.lower() and buffer.dtype != torch.float32:
            logger.warning(f"转换anchor buffer {name} 从 {buffer.dtype} 到 float32")
            buffer.data = buffer.data.to(torch.float32)
    
    return model

def main():
    """主函数"""
    # 设置参数
    model_yaml = os.path.join(YOLO_PATH, "ultralytics/cfg/models/v12/yolo12-A2C2f-DYT-EfficientHead-MambaOut.yaml")
    batch_size = 2
    img_size = 640
    
    # 检查文件是否存在
    if not os.path.exists(model_yaml):
        logger.error(f"模型配置文件不存在: {model_yaml}")
        sys.exit(1)
    
    # 加载数据配置(为了获取nc)
    data_yaml = "tomato_data.yaml"
    try:
        with open(data_yaml, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        logger.info(f"成功加载数据配置: {data_yaml}")
        nc = data_config.get("nc", 6)  # 默认为6个类别
    except Exception as e:
        logger.warning(f"加载数据配置失败，使用默认6类: {e}")
        nc = 6
    
    # 加载模型
    try:
        logger.info(f"加载模型: {model_yaml}")
        model = TomatoDetectionModel(cfg=model_yaml, nc=nc)
        logger.info("模型加载成功")
        logger.info(f"模型类型: {type(model)}")
        
        # 修复模型数据类型问题
        model = fix_model_dtype_issue(model)
        
        # 打印模型的探测头信息
        if hasattr(model, 'model') and hasattr(model.model, 'detect'):
            detect = model.model.detect
            logger.info(f"探测头类型: {type(detect)}")
            logger.info(f"探测头结构: {detect}")
            logger.info(f"探测头输入大小: {detect.args.imgsz if hasattr(detect, 'args') else 'Unknown'}")
        else:
            logger.warning("未找到模型探测头")
        
        # 准备模拟输入数据
        dummy_input = torch.randn(batch_size, 3, img_size, img_size)
        logger.info(f"输入形状: {dummy_input.shape}")
        
        # 设置为评估模式
        model.eval()
        
        # 使用try-except捕获可能的错误
        try:
            # 执行前向传播
            with torch.no_grad():
                logger.info("开始执行前向传播...")
                outputs = model(dummy_input)
                logger.info("前向传播成功")
                
                # 打印输出结构
                logger.info("=" * 50)
                logger.info("模型前向传播输出结构:")
                logger.info("=" * 50)
                
                if isinstance(outputs, (tuple, list)):
                    logger.info(f"输出是{type(outputs).__name__}，长度: {len(outputs)}")
                    for i, out in enumerate(outputs):
                        print_tensor_info(f"输出[{i}]", out)
                elif isinstance(outputs, dict):
                    logger.info(f"输出是字典，键: {list(outputs.keys())}")
                    for k, v in outputs.items():
                        print_tensor_info(f"输出['{k}']", v)
                else:
                    print_tensor_info("输出", outputs)
                
                # 特别检查pred_score/pred_distri/pred_rank
                logger.info("=" * 50)
                logger.info("检查特定输出字段:")
                logger.info("=" * 50)
                
                fields_to_check = [
                    "pred_score", "pred_distri", "pred_rank", 
                    "box_dist", "cls_dist", "dfl_dist",
                    "pred_boxes", "pred_classes"
                ]
                
                if isinstance(outputs, dict):
                    for field in fields_to_check:
                        if field in outputs:
                            print_tensor_info(field, outputs[field])
                        else:
                            logger.info(f"{field}: 不存在")
                
                # 尝试获取探测头的原始输出
                try:
                    if hasattr(model, 'forward_head'):
                        logger.info("=" * 50)
                        logger.info("探测头原始输出:")
                        logger.info("=" * 50)
                        head_outputs = model.forward_head(dummy_input)
                        if isinstance(head_outputs, (tuple, list)):
                            logger.info(f"探测头输出是{type(head_outputs).__name__}，长度: {len(head_outputs)}")
                            for i, out in enumerate(head_outputs):
                                print_tensor_info(f"探测头输出[{i}]", out)
                        else:
                            print_tensor_info("探测头输出", head_outputs)
                except Exception as e:
                    logger.error(f"获取探测头输出失败: {e}")
                
        except Exception as e:
            logger.error(f"前向传播失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 分析错误并尝试解决
            error_msg = str(e)
            if "expected scalar type Byte but found Float" in error_msg:
                logger.info("检测到数据类型不匹配错误，尝试定位...")
                
                # 检查anchor_grid相关buffer
                for name, buffer in model.named_buffers():
                    if "anchor" in name.lower():
                        logger.info(f"Buffer {name}: 类型={buffer.dtype}, 形状={buffer.shape}")
                        # 尝试转换为float32
                        if buffer.dtype != torch.float32:
                            logger.info(f"尝试将 {name} 转换为float32")
                            buffer.data = buffer.data.to(torch.float32)
                
                # 再次尝试前向传播
                try:
                    logger.info("再次尝试前向传播...")
                    with torch.no_grad():
                        outputs = model(dummy_input)
                    logger.info("前向传播成功")
                except Exception as e2:
                    logger.error(f"再次尝试前向传播失败: {e2}")
        
    except Exception as e:
        logger.error(f"调试过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
    logger.info("调试完成") 