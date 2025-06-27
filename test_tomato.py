#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试番茄检测模型的脚本
"""

from ultralytics import YOLO
from ultralytics.cfg import TASK2DATA, TASKS
import sys

# 打印配置信息
print("TASKS:", TASKS)
print("TASK2DATA:", TASK2DATA)

# 检查TASKS和TASK2DATA是否包含tomato
if "tomato" not in TASKS:
    print("错误：TASKS中未包含'tomato'任务")
    sys.exit(1)

if "tomato" not in TASK2DATA:
    print("错误：TASK2DATA中未包含'tomato'任务")
    sys.exit(1)

print("成功：TASKS和TASK2DATA中已包含'tomato'任务")

# 加载模型
try:
    # 创建一个新的模型
    model = YOLO("/root/shared-nvme/yolov12-main/yolov12-new/ultralytics/cfg/models/v12/yolo12-A2C2f-DYT-EfficientHead-MambaOut.yaml")
    print(f"模型类型: {type(model)}")
    print(f"模型任务: {model.task}")
    print("成功：创建番茄检测模型")
    
    # 测试能否正确加载数据路径
    train_args = model.train() if hasattr(model, "train") else {"data": None}
    if 'data' in train_args and train_args['data'] is not None:
        print(f"训练数据路径: {train_args['data']}")
    
    print("测试完成")
except Exception as e:
    print(f"错误: {str(e)}")
    raise e 