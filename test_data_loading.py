#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试脚本：诊断番茄数据集加载过程中instances丢失的问题
"""

import os
import sys
import logging
import torch
import numpy as np
from pathlib import Path

# 配置日志 - 确保输出到控制台
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data_loading_test.log")  # 同时保存到文件
    ]
)

logger = logging.getLogger("data_loading_test")
logger.info("脚本开始执行")

try:
    # 添加项目根目录到路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"当前目录: {current_dir}")
    sys.path.append(current_dir)
    
    from ultralytics.data.dataset import TomatoYOLODataset
    from ultralytics.data.augment import Compose, Mosaic, RandomPerspective, MixUp, RandomHSV, RandomFlip, Format, TomatoFormat, LetterBox
    from ultralytics.utils import LOGGER, IterableSimpleNamespace
    from ultralytics.utils.instance import Instances
    
    logger.info("所有模块导入成功")
except ImportError as e:
    logger.error(f"导入模块时出错: {e}")
    sys.exit(1)

# 创建一个自定义的Transform类，用于跟踪数据流
class TrackingTransform:
    def __init__(self, transform, name):
        self.transform = transform
        self.name = name
    
    def __call__(self, labels):
        logger.info(f"===== 进入 {self.name} =====")
        logger.info(f"输入标签类型: {type(labels)}")
        logger.info(f"输入标签键: {list(labels.keys())}")
        
        if "instances" in labels:
            logger.info(f"输入instances类型: {type(labels['instances'])}")
            logger.info(f"输入instances长度: {len(labels['instances'])}")
            
            # 检查instances的属性
            instances = labels["instances"]
            if hasattr(instances, "cluster_ids"):
                logger.info(f"instances.cluster_ids: {instances.cluster_ids is not None}")
            if hasattr(instances, "h_rel"):
                logger.info(f"instances.h_rel: {instances.h_rel is not None}")
        else:
            logger.warning(f"输入标签中没有instances字段!")
        
        # 应用变换
        try:
            result = self.transform(labels)
            logger.info(f"变换 {self.name} 成功应用")
        except Exception as e:
            logger.error(f"应用变换 {self.name} 时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # 返回原始标签，确保流程可以继续
            return labels
        
        logger.info(f"===== 离开 {self.name} =====")
        logger.info(f"输出标签类型: {type(result)}")
        logger.info(f"输出标签键: {list(result.keys())}")
        
        if "instances" in result:
            logger.info(f"输出instances类型: {type(result['instances'])}")
            logger.info(f"输出instances长度: {len(result['instances'])}")
        else:
            logger.warning(f"输出标签中没有instances字段!")
        
        return result

# 创建一个监控版的Compose
class MonitoringCompose(Compose):
    def __call__(self, data):
        logger.info(f"MonitoringCompose开始处理，共有 {len(self.transforms)} 个变换")
        
        for i, t in enumerate(self.transforms):
            logger.info(f"应用变换 {i}: {t.__class__.__name__}")
            if "instances" in data:
                logger.info(f"变换前 instances 存在，长度: {len(data['instances'])}")
            else:
                logger.warning(f"变换前 instances 不存在!")
            
            try:
                data = t(data)
                logger.info(f"变换 {i} 应用成功")
            except Exception as e:
                logger.error(f"应用变换 {i} 时出错: {e}")
                import traceback
                logger.error(traceback.format_exc())
            
            if "instances" in data:
                logger.info(f"变换后 instances 存在，长度: {len(data['instances'])}")
            else:
                logger.warning(f"变换后 instances 不存在!")
        
        return data

def test_dataset_loading():
    """测试番茄数据集加载过程"""
    logger.info("开始测试番茄数据集加载过程")
    
    # 打印当前工作目录
    logger.info(f"当前工作目录: {os.getcwd()}")
    
    # 创建数据集配置
    data = {
        "path": "D:/TomatoDataset",
        "train": "roboflow-v2/train/images",
        "val": "roboflow-v2/valid/images",
        "nc": 6,
        "names": ["fully_ripe", "ripe", "turning", "green", "background", "bunch"],
        "has_cluster_ids": True,
        "has_h_rel": True
    }
    
    # 检查数据路径是否存在
    train_path = os.path.join(data["path"], data["train"])
    logger.info(f"训练数据路径: {train_path}")
    if not os.path.exists(train_path):
        logger.error(f"训练数据路径不存在: {train_path}")
        return
    
    # 创建超参数
    hyp = IterableSimpleNamespace()
    hyp.mosaic = 1.0
    hyp.copy_paste = 0.0
    hyp.copy_paste_mode = "flip"
    hyp.mixup = 0.0
    hyp.degrees = 0.0
    hyp.translate = 0.1
    hyp.scale = 0.5
    hyp.shear = 0.0
    hyp.perspective = 0.0
    hyp.flipud = 0.0
    hyp.fliplr = 0.0
    hyp.hsv_h = 0.015
    hyp.hsv_s = 0.7
    hyp.hsv_v = 0.4
    hyp.mask_ratio = 4
    hyp.overlap_mask = True
    hyp.bgr = 0.0
    
    logger.info("超参数配置完成")
    
    # 创建数据集
    try:
        logger.info("开始创建TomatoYOLODataset")
        dataset = TomatoYOLODataset(
            img_path=train_path,
            imgsz=640,
            batch_size=16,
            augment=True,
            hyp=hyp,
            rect=False,
            cache=False,
            single_cls=False,
            stride=32,
            pad=0.0,
            prefix="train: ",
            data=data
        )
        logger.info(f"数据集创建成功，包含 {len(dataset)} 个样本")
    except Exception as e:
        logger.error(f"创建数据集时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # 检查数据集标签
    logger.info("检查数据集标签格式")
    sample_idx = 0
    try:
        if sample_idx >= len(dataset.labels):
            logger.error(f"样本索引 {sample_idx} 超出范围，数据集只有 {len(dataset.labels)} 个样本")
            sample_idx = 0
            
        label = dataset.labels[sample_idx]
        logger.info(f"样本 {sample_idx} 标签类型: {type(label)}")
        logger.info(f"样本 {sample_idx} 标签键: {list(label.keys())}")
        
        # 检查是否有cluster_ids和h_rel
        if "cluster_ids" in label:
            logger.info(f"样本 {sample_idx} cluster_ids 形状: {label['cluster_ids'].shape}")
        if "h_rel" in label:
            logger.info(f"样本 {sample_idx} h_rel 形状: {label['h_rel'].shape}")
    except Exception as e:
        logger.error(f"检查标签时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # 创建监控版的数据增强管道
    logger.info("创建监控版的数据增强管道")
    
    try:
        # 创建基本变换
        logger.info("创建基本变换")
        mosaic = TrackingTransform(Mosaic(dataset, imgsz=640, p=hyp.mosaic), "Mosaic")
        letter_box = TrackingTransform(LetterBox(new_shape=(640, 640)), "LetterBox")
        affine = TrackingTransform(RandomPerspective(
            degrees=hyp.degrees,
            translate=hyp.translate,
            scale=hyp.scale,
            shear=hyp.shear,
            perspective=hyp.perspective,
            pre_transform=letter_box
        ), "RandomPerspective")
        
        # 创建预变换
        logger.info("创建预变换")
        pre_transform = Compose([mosaic, affine])
        pre_transform = TrackingTransform(pre_transform, "PreTransform")
        
        # 创建MixUp变换
        logger.info("创建MixUp变换")
        mixup = TrackingTransform(MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup), "MixUp")
        
        # 创建HSV变换
        logger.info("创建HSV变换")
        hsv = TrackingTransform(RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v), "RandomHSV")
        
        # 创建翻转变换
        logger.info("创建翻转变换")
        flip_ud = TrackingTransform(RandomFlip(direction="vertical", p=hyp.flipud), "RandomFlipUD")
        flip_lr = TrackingTransform(RandomFlip(direction="horizontal", p=hyp.fliplr), "RandomFlipLR")
        
        # 创建格式化变换
        logger.info("创建格式化变换")
        formatter = TrackingTransform(TomatoFormat(
            bbox_format="xywh",
            normalize=True,
            return_mask=False,
            return_keypoint=False,
            return_obb=False,
        ), "TomatoFormat")
        
        # 组合所有变换
        logger.info("组合所有变换")
        transforms = MonitoringCompose([
            pre_transform,
            mixup,
            hsv,
            flip_ud,
            flip_lr,
            formatter
        ])
        
        logger.info("数据增强管道创建成功")
    except Exception as e:
        logger.error(f"创建数据增强管道时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # 测试数据加载和增强
    logger.info("测试数据加载和增强")
    try:
        # 获取原始标签
        logger.info(f"获取样本 {sample_idx} 的原始标签")
        raw_label = dataset.get_image_and_label(sample_idx)
        logger.info(f"原始标签类型: {type(raw_label)}")
        logger.info(f"原始标签键: {list(raw_label.keys())}")
        
        if "instances" in raw_label:
            logger.info(f"原始instances类型: {type(raw_label['instances'])}")
            logger.info(f"原始instances长度: {len(raw_label['instances'])}")
            
            # 检查instances的属性
            instances = raw_label["instances"]
            if hasattr(instances, "cluster_ids"):
                logger.info(f"原始instances.cluster_ids: {instances.cluster_ids is not None}")
            if hasattr(instances, "h_rel"):
                logger.info(f"原始instances.h_rel: {instances.h_rel is not None}")
        else:
            logger.warning("原始标签中没有instances字段!")
        
        # 应用变换
        logger.info("开始应用数据增强变换")
        transformed = transforms(raw_label)
        logger.info(f"变换后标签类型: {type(transformed)}")
        logger.info(f"变换后标签键: {list(transformed.keys())}")
        
        # 检查最终结果是否有cluster_ids和h_rel
        if "cluster_ids" in transformed:
            logger.info(f"变换后 cluster_ids 形状: {transformed['cluster_ids'].shape}")
        else:
            logger.warning("变换后标签中没有cluster_ids字段!")
            
        if "h_rel" in transformed:
            logger.info(f"变换后 h_rel 形状: {transformed['h_rel'].shape}")
        else:
            logger.warning("变换后标签中没有h_rel字段!")
    except Exception as e:
        logger.error(f"测试数据加载和增强时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # 测试__getitem__方法
    logger.info("测试__getitem__方法")
    try:
        item = dataset[sample_idx]
        logger.info(f"__getitem__返回类型: {type(item)}")
        logger.info(f"__getitem__返回键: {list(item.keys())}")
        
        # 检查是否有cluster_ids和h_rel
        if "cluster_ids" in item:
            logger.info(f"__getitem__ cluster_ids 形状: {item['cluster_ids'].shape}")
        else:
            logger.warning("__getitem__返回中没有cluster_ids字段!")
            
        if "h_rel" in item:
            logger.info(f"__getitem__ h_rel 形状: {item['h_rel'].shape}")
        else:
            logger.warning("__getitem__返回中没有h_rel字段!")
    except Exception as e:
        logger.error(f"测试__getitem__方法时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("测试完成")

if __name__ == "__main__":
    try:
        logger.info("主函数开始执行")
        test_dataset_loading()
        logger.info("主函数执行完成")
    except Exception as e:
        logger.error(f"主函数执行出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        logger.info("脚本执行结束") 