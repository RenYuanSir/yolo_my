import argparse
import sys
from pathlib import Path
import logging

# 配置更详细的日志输出
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler('train_debug.log')
                    ])

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(model=r'/root/shared-nvme/yolov12-main/yolov12-new/ultralytics/cfg/models/v12/yolo12s-A2C2f-DYT-EfficientHead-MambaOut.yaml')
    ## model.load('yolov12s.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model.train(data=r'/root/shared-nvme/tomato_transStyle/data.yaml',
                imgsz=640,
                epochs=5,  # 先运行少量epoch测试修复效果
                batch=16,  # 减小batch size以减轻显存压力
                workers=8,
                device=[0,],
                optimizer='SGD',
                close_mosaic=0,  # 从一开始就关闭镶嵌增强，减少复杂性
                project='runs/train',
                name='debug',
                single_cls=False,
                lr0=0.01,
                cache=False,
                loss='TomatoDetectWithRankLoss(lambda_rank=0.2)',
                verbose=True,  # 启用详细输出
                hsv_h=0.0, # 关闭HSV色调增强
                hsv_s=0.0, # 关闭HSV饱和度增强
                hsv_v=0.0, # 关闭HSV亮度增强
                degrees=0.0, # 关闭旋转
                translate=0.0, # 关闭平移
                scale=0.0, # 关闭缩放
                shear=0.0, # 关闭剪切
                perspective=0.0005, # 关闭透视变换
                flipud=0.0, # 关闭上下翻转
                fliplr=0.0, # 关闭左右翻转
                mosaic=0.0, # 关闭拼接
                mixup=0.0 # 关闭混合
                ) 