import argparse
import sys
from pathlib import Path
import logging
import torch


logger = logging.getLogger("TrainDebug")

from ultralytics import YOLO

if __name__ == '__main__':
    try:
        # 禁用图像验证，提高容错性
        torch.set_float32_matmul_precision('high')
        
        logger.info("初始化YOLO模型")
        model = YOLO(model=r'/root/shared-nvme/yolov12-main/yolov12-new/ultralytics/cfg/models/v12/yolo12s-A2C2f-DYT-EfficientHead-MambaOut.yaml')
        
        # 极简训练选项
        logger.info("开始训练")
        model.train(data=r'/root/shared-nvme/tomato_transStyle/data.yaml',
                    imgsz=640,
                    epochs=1,  # 只训练一个epoch用于测试
                    batch=4,   # 使用更小的batch以降低资源需求
                    workers=1, # 减少worker数量，避免多线程问题
                    device=[0,],
                    optimizer='SGD',
                    project='runs/train',
                    name='debug',
                    single_cls=False,
                    lr0=0.01,
                    cache=False,
                    loss='TomatoDetectWithRankLoss(lambda_rank=0.2)',
                    verbose=True,
                    # 完全关闭所有数据增强
                    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                    degrees=0.0, translate=0.0, scale=0.0, shear=0.0,
                    perspective=0.0, flipud=0.0, fliplr=0.0,
                    mosaic=0.0, mixup=0.0, close_mosaic=0
                    )
    except Exception as e:
        logger.error(f"训练失败: {e}")
        import traceback
        logger.error(traceback.format_exc()) 