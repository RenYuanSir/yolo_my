#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ultralytics.models.yolo.tomato.val import TomatoValidator
import torch
import torch.nn as nn

class DummyHead(nn.Module):
    pass

def main():
    # 创建模拟模型
    model = torch.nn.Module()
    head = DummyHead()
    head.stride = [8, 16, 32]
    head.reg_max = 16
    head.nc = 2
    head.no = 70
    model.model = [nn.Module(), head]
    
    # 创建模拟输入
    preds = {
        'features': [
            torch.rand(2, 70, 84, 84),
            torch.rand(2, 70, 42, 42),
            torch.rand(2, 70, 21, 21)
        ],
        'h_pos': [
            torch.rand(2, 1, 84, 84),
            torch.rand(2, 1, 42, 42),
            torch.rand(2, 1, 21, 21)
        ]
    }
    
    # 创建验证器
    validator = TomatoValidator(args={
        'device': 'cpu',
        'conf': 0.25,
        'iou': 0.45,
        'max_det': 300,
        'single_cls': False,
        'agnostic_nms': False
    })
    validator.model = model
    validator.batch = {'img': torch.rand(2, 3, 640, 640)}
    validator.lb = []
    
    # 测试normalize_predictions
    print('测试normalize_predictions:')
    feats, h_pos = validator._normalize_predictions(preds)
    
    # 测试extract_predictions
    print('\n测试extract_predictions:')
    distri, scores = validator._extract_predictions(feats)
    print(f'分布形状: {distri.shape}, 分数形状: {scores.shape}')
    
    # 测试完整的后处理流程
    print('\n测试完整后处理流程:')
    processed = validator.postprocess(preds)
    print(f'后处理结果: {[p.shape if len(p) > 0 else 0 for p in processed]}')
    
    print('\n测试完成')

if __name__ == "__main__":
    main() 