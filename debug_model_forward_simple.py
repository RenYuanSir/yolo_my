#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import yaml

# 添加YOLO路径
YOLO_PATH = os.path.dirname(os.path.abspath(__file__))
print(f"YOLO_PATH: {YOLO_PATH}")
sys.path.insert(0, YOLO_PATH)  # 确保本地路径优先

# 导入YOLO相关模块
from ultralytics.nn.tasks import TomatoDetectionModel

# 解决数据类型问题的函数
def fix_model_dtype(model):
    """修复模型中的数据类型问题"""
    print("尝试修复模型数据类型问题...")
    # 强制转换所有buffer为float32
    for name, buffer in model.named_buffers():
        if buffer.dtype != torch.float32:
            print(f"转换buffer {name} 从 {buffer.dtype} 到 float32")
            buffer.data = buffer.data.to(torch.float32)
    return model

# 打印张量信息的函数
def print_tensor_info(name, tensor, indent=0):
    """打印张量的详细信息"""
    indent_str = '  ' * indent
    if isinstance(tensor, torch.Tensor):
        print(f"{indent_str}{name}:")
        print(f"{indent_str}- 形状: {tensor.shape}")
        print(f"{indent_str}- 数据类型: {tensor.dtype}")
        print(f"{indent_str}- 设备: {tensor.device}")
        try:
            print(f"{indent_str}- 值范围: [{tensor.min().item():.6f}, {tensor.max().item():.6f}]")
        except:
            print(f"{indent_str}- 值范围: 无法计算")
    elif isinstance(tensor, (list, tuple)):
        print(f"{indent_str}{name}: 是列表/元组，长度={len(tensor)}")
        for i, item in enumerate(tensor):
            print_tensor_info(f"{name}[{i}]", item, indent + 1)
    elif isinstance(tensor, dict):
        print(f"{indent_str}{name}: 是字典，键={list(tensor.keys())}")
        for k, v in tensor.items():
            print_tensor_info(f"{name}['{k}']", v, indent + 1)
    else:
        print(f"{indent_str}{name}: 类型={type(tensor)}")

def main():
    # 设置参数
    model_yaml = os.path.join(YOLO_PATH, "ultralytics/cfg/models/v12/yolo12-A2C2f-DYT-EfficientHead-MambaOut.yaml")
    batch_size = 2
    img_size = 640
    
    # 加载数据配置(为了获取nc)
    data_yaml = "tomato_data.yaml"
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    print(f"成功加载数据配置: {data_yaml}")
    nc = data_config.get("nc", 6)  # 默认为6个类别
    
    # 加载模型
    print(f"加载模型: {model_yaml}")
    model = TomatoDetectionModel(cfg=model_yaml, nc=nc)
    print("模型加载成功")
    
    # 修复模型数据类型问题
    model = fix_model_dtype(model)
    
    # 准备模拟输入数据
    dummy_input = torch.randn(batch_size, 3, img_size, img_size)
    print(f"输入形状: {dummy_input.shape}")
    
    # 设置为评估模式
    model.eval()
    
    # 执行前向传播
    with torch.no_grad():
        print("执行前向传播...")
        outputs = model(dummy_input)
        print("前向传播成功")
        
        # 打印输出结构
        print("=" * 50)
        print("模型前向传播输出结构:")
        print("=" * 50)
        
        if isinstance(outputs, (tuple, list)):
            print(f"输出是{type(outputs).__name__}，长度: {len(outputs)}")
            for i, out in enumerate(outputs):
                print_tensor_info(f"输出[{i}]", out)
        elif isinstance(outputs, dict):
            print(f"输出是字典，键: {list(outputs.keys())}")
            for k, v in outputs.items():
                print_tensor_info(f"输出['{k}']", v)
        else:
            print_tensor_info("输出", outputs)
        
        # 特别检查pred_score/pred_distri/pred_rank
        print("=" * 50)
        print("检查特定输出字段:")
        print("=" * 50)
        
        fields_to_check = [
            "pred_score", "pred_distri", "pred_rank", 
            "box_dist", "cls_dist", "dfl_dist",
            "pred_boxes", "pred_classes", "h_pos"
        ]
        
        # 尝试在输出中查找特定字段
        if isinstance(outputs, tuple) and len(outputs) > 1 and isinstance(outputs[1], dict):
            for field in fields_to_check:
                if field in outputs[1]:
                    print(f"{field} 在 outputs[1] 中:")
                    print_tensor_info(field, outputs[1][field])
                else:
                    print(f"{field}: 不存在于 outputs[1] 中")

if __name__ == "__main__":
    main()
    print("调试完成") 