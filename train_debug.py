import argparse
import sys
import os
from pathlib import Path
import logging
import torch
import traceback

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrainDebug")

from ultralytics import YOLO
from ultralytics.utils import LOGGER, callbacks

# 导入特定的数据集和格式化类
from ultralytics.data.dataset import TomatoYOLODataset
from ultralytics.data.augment import TomatoFormat

# 注册训练钩子
def on_train_start(trainer):
    """训练开始时的回调函数"""
    logger.info("训练开始")
    if hasattr(trainer, 'train_loader'):
        logger.info("添加训练数据加载器调试钩子")
        add_data_debug_hooks(trainer.train_loader)
    if hasattr(trainer, 'val_loader'):
        logger.info("添加验证数据加载器调试钩子")
        add_data_debug_hooks(trainer.val_loader)
        
# 注册批次回调
def on_train_batch_start(trainer):
    """每个批次开始时的回调函数"""
    if trainer.epoch == 0 and trainer.batch_idx < 5:  # 只记录前5个批次
        logger.info(f"epoch {trainer.epoch}, batch {trainer.batch_idx} 开始")
        if hasattr(trainer, 'batch'):
            batch = trainer.batch
            logger.info(f"批次字段: {list(batch.keys()) if isinstance(batch, dict) else '非字典类型'}")
            if isinstance(batch, dict):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        logger.info(f"  {k}: 形状={v.shape}, 类型={v.dtype}, 设备={v.device}")
                        # 检查nan和inf
                        if v.dtype.is_floating_point:
                            nan_count = torch.isnan(v).sum().item()
                            inf_count = torch.isinf(v).sum().item()
                            if nan_count > 0 or inf_count > 0:
                                logger.warning(f"  {k}包含 {nan_count} 个NaN和 {inf_count} 个Inf")

# 添加钩子监控数据加载过程
def add_data_debug_hooks(dataloader):
    """添加钩子函数来跟踪数据集加载过程"""
    if not hasattr(dataloader.dataset, '_old_getitem'):
        # 保存原始__getitem__方法
        dataloader.dataset._old_getitem = dataloader.dataset.__getitem__
        
        # 定义新的__getitem__方法
        def new_getitem(self, index):
            try:
                item = self._old_getitem(index)
                # 记录成功加载的项
                if index % 10 == 0:  # 只记录每10个样本，避免日志过多
                    logger.debug(f"成功加载索引 {index}, 字段: {list(item.keys())}")
                    if 'cls' in item and hasattr(item['cls'], 'shape'):
                        logger.debug(f"cls形状: {item['cls'].shape}")
                    if 'bboxes' in item and hasattr(item['bboxes'], 'shape'):
                        logger.debug(f"bboxes形状: {item['bboxes'].shape}")
                    if 'cluster_ids' in item and hasattr(item['cluster_ids'], 'shape'):
                        logger.debug(f"cluster_ids形状: {item['cluster_ids'].shape}")
                    if 'h_rel' in item and hasattr(item['h_rel'], 'shape'):
                        logger.debug(f"h_rel形状: {item['h_rel'].shape}")
                        
                return item
            except Exception as e:
                # 记录错误
                logger.error(f"加载索引 {index} 出错: {e}")
                logger.error(traceback.format_exc())
                raise
                
        # 替换__getitem__方法
        dataloader.dataset.__getitem__ = lambda idx: new_getitem(dataloader.dataset, idx)
        logger.info("已添加数据集调试钩子")

if __name__ == '__main__':
    try:
        # 解析命令行参数
        parser = argparse.ArgumentParser()
        parser.add_argument('--use_original', action='store_true', help='使用原始YOLODataset而不是TomatoYOLODataset')
        parser.add_argument('--debug_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help='日志级别')
        args = parser.parse_args()
        
        # 设置日志级别
        if args.debug_level == 'DEBUG':
            LOGGER.setLevel(logging.DEBUG)
            logger.setLevel(logging.DEBUG)
        elif args.debug_level == 'INFO':
            LOGGER.setLevel(logging.INFO)
            logger.setLevel(logging.INFO)
        elif args.debug_level == 'WARNING':
            LOGGER.setLevel(logging.WARNING)
            logger.setLevel(logging.WARNING)
        elif args.debug_level == 'ERROR':
            LOGGER.setLevel(logging.ERROR)
            logger.setLevel(logging.ERROR)
            
        # 禁用图像验证，提高容错性
        torch.set_float32_matmul_precision('high')
        
        # 记录系统信息
        logger.info(f"Python版本: {sys.version}")
        logger.info(f"PyTorch版本: {torch.__version__}")
        logger.info(f"CUDA是否可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA设备: {torch.cuda.get_device_name(0)}")
        
        logger.info("初始化YOLO模型")
        model = YOLO(model=r'D:\yoloProject\yolo_new_local\yolo_my-0.1\ultralytics\cfg\models\v12\yolov12.yaml')
        
        # 极简训练选项
        logger.info("开始训练")
        try:
            # 如果使用原始YOLODataset，需要修改环境变量
            if args.use_original:
                logger.info("使用原始YOLODataset")
                import os
                os.environ["USE_ORIGINAL_DATASET"] = "1"
            else:
                logger.info("使用TomatoYOLODataset")
                
            # 注册回调函数
            # 正确方式是使用callbacks模块的on_*函数
            callbacks.on_train_start = on_train_start
            callbacks.on_train_batch_start = on_train_batch_start
                
            # 训练
            model.train(data=r"D:\TomatoDataset\v2-test_mini\data.yaml",
                    imgsz=640,
                    epochs=100,  # 只训练一个epoch用于测试
                    batch=4,   # 使用更小的batch以降低资源需求
                    workers=0,  # 减少worker数量，避免多线程问题
                    device=[0,],
                    optimizer='SGD',
                    project='runs/train',
                    name='debug',
                    single_cls=False,
                    lr0=0.0005,
                    cache=False,
                    loss='TomatoDetectWithRankLoss(lambda_rank=0.5)',
                    verbose=True,
                    task='tomato',  # 明确指定使用tomato任务，确保使用TomatoYOLODataset
                    amp=False,      # 禁用自动混合精度训练，避免梯度缩放器错误
                    # 完全关闭所有数据增强
                    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                    degrees=0.0, translate=0.0, scale=0.0, shear=0.0,
                    perspective=0.0, flipud=0.0, fliplr=0.0,
                    mosaic=0.0, mixup=0.0, close_mosaic=0
                    )
            logger.info("训练完成")
        except Exception as e:
            logger.error(f"训练过程中出错: {e}")
            logger.error(traceback.format_exc())
            
    except Exception as e:
        logger.error(f"训练失败: {e}")
        logger.error(traceback.format_exc()) 