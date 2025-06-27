#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch

# 添加路径
YOLO_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, YOLO_PATH)

# 导入模型
from ultralytics.nn.tasks import TomatoDetectionModel
import yaml

# 修复数据类型
def fix_dtype(model):
    for name, buffer in model.named_buffers():
        if buffer.dtype != torch.float32:
            print(f"转换 {name} 从 {buffer.dtype} 到 float32")
            buffer.data = buffer.data.to(torch.float32)
    return model

# 输出张量信息
def print_output(name, tensor):
    if isinstance(tensor, torch.Tensor):
        print(f"{name}: 形状={tensor.shape}, 类型={tensor.dtype}")
    elif isinstance(tensor, (list, tuple)):
        print(f"{name}: 是{type(tensor).__name__}, 长度={len(tensor)}")
        for i, item in enumerate(tensor):
            print_output(f"{name}[{i}]", item)
    elif isinstance(tensor, dict):
        print(f"{name}: 是字典, 键={list(tensor.keys())}")
        for k, v in tensor.items():
            print_output(f"{name}['{k}']", v)
    else:
        print(f"{name}: 类型={type(tensor)}")

# 主函数
nc = 6  # 类别数量
model_yaml = os.path.join(YOLO_PATH, "ultralytics/cfg/models/v12/yolo12-A2C2f-DYT-EfficientHead-MambaOut.yaml")
    
print("加载模型...")
model = TomatoDetectionModel(cfg=model_yaml, nc=nc)
model = fix_dtype(model)
model.eval()

print("执行前向传播...")
dummy_input = torch.randn(2, 3, 640, 640)
with torch.no_grad():
    outputs = model(dummy_input)
    
print("\n模型前向传播输出结构:")
print_output("outputs", outputs) 