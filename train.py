import argparse
import sys
from pathlib import Path

# Add project root to sys.path to ensure ultralytics can be imported
# This assumes train.py is in the project root
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(model=r'/root/shared-nvme/yolov12-main/yolov12-new/ultralytics/cfg/models/v12/yolo12s-A2C2f-DYT-EfficientHead-MambaOut.yaml')
    ## model.load('yolov12s.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model.train(data=r'/root/shared-nvme/tomato_transStyle/data.yaml',
                imgsz=640,
                epochs=300,
                batch=64,
                workers=16,
                optimizer='SGD',
                close_mosaic=30,
                project='runs/train',
                name='exp1',
                single_cls=False,
                lr0=0.01,
                cache=False,
                loss='TomatoDetectWithRankLoss(lambda_pos=0.3, lambda_rank=0.2, lambda_conf=0.4)',
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
                mosaic=0.5, # 关闭拼接
                mixup=0.0 # 关闭混合
                )