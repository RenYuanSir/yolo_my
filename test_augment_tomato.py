#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试番茄数据集的数据增强过程，检查是否正确保留cluster_ids和h_rel字段
"""

import os
import sys
import torch
import yaml
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from copy import deepcopy

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tomato_augment_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_augment_tomato")

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入必要的类
from ultralytics.data.dataset import TomatoYOLODataset
from ultralytics.data.augment import v8_transforms, Mosaic, RandomPerspective, TomatoFormat, Format, Compose
from ultralytics.utils import LOGGER
from ultralytics.utils.instance import Instances

# 创建模拟数据
def create_mock_data():
    """创建模拟数据用于测试"""
    # 创建一个简单的图像
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    img[100:300, 100:300, 0] = 255  # 添加一个大的红色方块
    img[350:550, 350:550, 1] = 255  # 添加一个大的绿色方块
    
    # 创建边界框 [cx, cy, w, h] 格式 - XYWH（中心点坐标+宽高）
    # 确保中心点在[0,1]范围内，宽高合理且大于0
    bboxes = np.array([
        [0.3, 0.3, 0.3, 0.3],    # 中心在(192, 192)，宽高约为192像素
        [0.7, 0.7, 0.3, 0.3],    # 中心在(448, 448)，宽高约为192像素
    ], dtype=np.float32)
    
    # 创建类别标签
    cls = np.array([0, 1], dtype=np.int64)
    
    # 创建cluster_ids - 两个不同的串
    cluster_ids = np.array([[1], [2]], dtype=np.float32)
    
    # 创建h_rel - 每个串内高度不同
    h_rel = np.array([[0.5], [0.7]], dtype=np.float32)
    
    # 创建Instances对象
    instances = Instances(bboxes=bboxes, normalized=True)  # 指定normalized=True，表示边界框已经在[0,1]范围内
    instances.cluster_ids = cluster_ids
    instances.h_rel = h_rel
    
    # 创建标签字典
    labels = {
        "img": img,
        "cls": cls,
        "instances": instances,
        "im_file": "mock_image.jpg",
        "ori_shape": img.shape[:2],
        "resized_shape": img.shape[:2],
        "batch_idx": np.array([0] * len(cls), dtype=np.float32),
    }
    
    return labels

# 创建模拟数据集
class MockTomatoDataset(Dataset):
    """模拟番茄数据集"""
    def __init__(self, num_samples=10):
        self.num_samples = num_samples
        self.has_cluster_ids = True
        self.has_h_rel = True
        self.use_segments = False
        self.use_keypoints = False
        self.use_obb = False
        self.data = {
            "has_cluster_ids": True,
            "has_h_rel": True,
            "names": ["Fully_Ripe", "Ripe", "Breaking", "Green", "Ripe_Bunch", "Unripe_Bunch"],
            "nc": 6
        }
        self.buffer = []  # 添加buffer属性
        self.max_buffer_length = 0  # 添加max_buffer_length属性
        self.stride = 32  # 添加stride属性
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        return create_mock_data()
        
    def get_image_and_label(self, idx):
        return create_mock_data()

class DebugTransform:
    """用于调试的变换，记录输入和输出"""
    def __init__(self, transform, name):
        self.transform = transform
        self.name = name
        
    def __call__(self, labels):
        logger.info(f"===== 开始应用变换: {self.name} =====")
        
        # 检查输入标签
        has_instances = "instances" in labels
        if has_instances:
            instances = labels["instances"]
            logger.info(f"输入 instances: 类型={type(instances)}, 长度={len(instances)}")
            has_cluster_ids = hasattr(instances, 'cluster_ids') and instances.cluster_ids is not None
            has_h_rel = hasattr(instances, 'h_rel') and instances.h_rel is not None
            logger.info(f"输入 has_cluster_ids={has_cluster_ids}, has_h_rel={has_h_rel}")
            
            if has_cluster_ids:
                logger.info(f"输入 cluster_ids: 类型={type(instances.cluster_ids)}, 形状={instances.cluster_ids.shape}")
            if has_h_rel:
                logger.info(f"输入 h_rel: 类型={type(instances.h_rel)}, 形状={instances.h_rel.shape}")
        else:
            logger.warning(f"输入标签中没有instances字段!")
        
        # 应用变换
        try:
            result = self.transform(deepcopy(labels))
            logger.info(f"变换 {self.name} 应用成功")
        except Exception as e:
            logger.error(f"应用变换 {self.name} 时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return labels
        
        # 检查输出标签
        has_instances = "instances" in result
        if has_instances:
            instances = result["instances"]
            logger.info(f"输出 instances: 类型={type(instances)}, 长度={len(instances)}")
            has_cluster_ids = hasattr(instances, 'cluster_ids') and instances.cluster_ids is not None
            has_h_rel = hasattr(instances, 'h_rel') and instances.h_rel is not None
            logger.info(f"输出 has_cluster_ids={has_cluster_ids}, has_h_rel={has_h_rel}")
            
            if has_cluster_ids:
                logger.info(f"输出 cluster_ids: 类型={type(instances.cluster_ids)}, 形状={instances.cluster_ids.shape}")
            if has_h_rel:
                logger.info(f"输出 h_rel: 类型={type(instances.h_rel)}, 形状={instances.h_rel.shape}")
        else:
            logger.warning(f"输出标签中没有instances字段!")
            
        # 检查输出中是否有cluster_ids和h_rel字段（非instances内部）
        if "cluster_ids" in result:
            logger.info(f"输出中有独立的cluster_ids字段: 类型={type(result['cluster_ids'])}, 形状={result['cluster_ids'].shape}")
        if "h_rel" in result:
            logger.info(f"输出中有独立的h_rel字段: 类型={type(result['h_rel'])}, 形状={result['h_rel'].shape}")
            
        logger.info(f"===== 完成变换: {self.name} =====")
        return result

class DebugCompose(Compose):
    """带调试信息的Compose类"""
    def __call__(self, data):
        logger.info(f"开始应用Compose，共有 {len(self.transforms)} 个变换")
        for i, t in enumerate(self.transforms):
            name = t.__class__.__name__
            logger.info(f"应用变换 {i}: {name}")
            
            # 检查输入
            if "instances" in data:
                instances = data["instances"]
                logger.info(f"变换 {name} 前: instances存在，长度={len(instances)}")
                if hasattr(instances, 'cluster_ids') and instances.cluster_ids is not None:
                    logger.info(f"变换 {name} 前: cluster_ids存在，形状={instances.cluster_ids.shape}")
                if hasattr(instances, 'h_rel') and instances.h_rel is not None:
                    logger.info(f"变换 {name} 前: h_rel存在，形状={instances.h_rel.shape}")
            else:
                logger.warning(f"变换 {name} 前: instances不存在!")
                
            # 应用变换
            try:
                data = t(data)
            except Exception as e:
                logger.error(f"应用变换 {i}:{name} 时出错: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
                
            # 检查输出
            if "instances" in data:
                instances = data["instances"]
                logger.info(f"变换 {name} 后: instances存在，长度={len(instances)}")
                if hasattr(instances, 'cluster_ids') and instances.cluster_ids is not None:
                    logger.info(f"变换 {name} 后: cluster_ids存在，形状={instances.cluster_ids.shape}")
                if hasattr(instances, 'h_rel') and instances.h_rel is not None:
                    logger.info(f"变换 {name} 后: h_rel存在，形状={instances.h_rel.shape}")
            else:
                logger.warning(f"变换 {name} 后: instances不存在!")
                
            # 检查是否有独立的cluster_ids和h_rel字段
            if "cluster_ids" in data:
                logger.info(f"变换 {name} 后: 独立的cluster_ids存在，形状={data['cluster_ids'].shape}")
            if "h_rel" in data:
                logger.info(f"变换 {name} 后: 独立的h_rel存在，形状={data['h_rel'].shape}")
                
        return data

def create_debug_transforms(dataset, imgsz):
    """创建带调试信息的变换"""
    # 仅使用RandomPerspective和TomatoFormat，跳过Mosaic
    perspective = DebugTransform(RandomPerspective(
        degrees=0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        pre_transform=None
    ), "RandomPerspective")
    
    # 使用TomatoFormat
    formatter = DebugTransform(TomatoFormat(
        bbox_format='xywh',
        normalize=True,
        return_mask=False,
        return_keypoint=False,
        return_obb=False,
    ), "TomatoFormat")
    
    # 创建调试版Compose
    transforms = DebugCompose([perspective, formatter])
    return transforms

def test_augment_tomato():
    """测试番茄数据集的数据增强过程"""
    logger.info("开始测试番茄数据集的数据增强过程")
    
    # 创建模拟数据集
    dataset = MockTomatoDataset(num_samples=10)
    logger.info(f"创建模拟数据集成功: {len(dataset)} 张图像")
    
    # 检查数据集属性
    logger.info(f"数据集属性: has_cluster_ids={dataset.has_cluster_ids}, has_h_rel={dataset.has_h_rel}")
    
    # 创建参数
    class Args:
        def __init__(self):
            self.imgsz = 640
            self.batch = 2
            self.rect = False
            self.cache = None
            self.single_cls = False
            self.augment = True
            self.stride = 32
            self.pad = 0.0
            self.loss = "TomatoDetectWithRankLoss"  # 确保启用cluster_ids和h_rel
            # 添加必要的超参数 - 关闭大部分增强
            self.mosaic = 0.0  # 关闭mosaic
            self.mixup = 0.0
            self.copy_paste = 0.0
            self.copy_paste_mode = "flip"
            self.mask_ratio = 4.0
            self.overlap_mask = True
            self.degrees = 0.0
            self.translate = 0.1
            self.scale = 0.5
            self.shear = 0.0
            self.perspective = 0.0
            self.flipud = 0.0
            self.fliplr = 0.0
            self.hsv_h = 0.0
            self.hsv_s = 0.0
            self.hsv_v = 0.0
    
    args = Args()
    
    try:
        # 创建调试版变换
        transforms = create_debug_transforms(dataset, args.imgsz)
        
        # 获取一个样本并应用变换
        sample_idx = 0
        sample = dataset.get_image_and_label(sample_idx)
        logger.info(f"获取样本 {sample_idx} 成功")
        
        # 检查原始样本
        if "instances" in sample:
            instances = sample["instances"]
            logger.info(f"原始样本 instances: 类型={type(instances)}, 长度={len(instances)}")
            has_cluster_ids = hasattr(instances, 'cluster_ids') and instances.cluster_ids is not None
            has_h_rel = hasattr(instances, 'h_rel') and instances.h_rel is not None
            logger.info(f"原始样本 has_cluster_ids={has_cluster_ids}, has_h_rel={has_h_rel}")
            
            if has_cluster_ids:
                logger.info(f"原始样本 cluster_ids: 类型={type(instances.cluster_ids)}, 形状={instances.cluster_ids.shape}")
            if has_h_rel:
                logger.info(f"原始样本 h_rel: 类型={type(instances.h_rel)}, 形状={instances.h_rel.shape}")
        else:
            logger.warning(f"原始样本中没有instances字段!")
        
        # 应用变换
        logger.info("开始应用变换...")
        transformed = transforms(sample)
        logger.info("变换应用完成")
        
        # 检查变换后的样本
        if "instances" in transformed:
            logger.info("变换后的样本包含instances字段")
        else:
            logger.warning("变换后的样本不包含instances字段!")
            
        if "cluster_ids" in transformed:
            logger.info(f"变换后的样本包含独立的cluster_ids字段: 形状={transformed['cluster_ids'].shape}")
        else:
            logger.warning("变换后的样本不包含独立的cluster_ids字段!")
            
        if "h_rel" in transformed:
            logger.info(f"变换后的样本包含独立的h_rel字段: 形状={transformed['h_rel'].shape}")
        else:
            logger.warning("变换后的样本不包含独立的h_rel字段!")
        
        # 检查变换后的样本中的字段是否与原始样本一致
        logger.info(f"变换后的样本字段: {list(transformed.keys())}")
        
        # 直接测试TomatoFormat
        logger.info("\n\n直接测试TomatoFormat...")
        tomato_format = TomatoFormat(
            bbox_format='xywh',
            normalize=True,
            return_mask=False,
            return_keypoint=False,
            return_obb=False,
        )
        
        # 获取一个新样本
        new_sample = dataset.get_image_and_label(0)
        logger.info("获取新样本成功")
        
        # 检查新样本
        if "instances" in new_sample:
            instances = new_sample["instances"]
            logger.info(f"新样本 instances: 类型={type(instances)}, 长度={len(instances)}")
            has_cluster_ids = hasattr(instances, 'cluster_ids') and instances.cluster_ids is not None
            has_h_rel = hasattr(instances, 'h_rel') and instances.h_rel is not None
            logger.info(f"新样本 has_cluster_ids={has_cluster_ids}, has_h_rel={has_h_rel}")
            
            if has_cluster_ids:
                logger.info(f"新样本 cluster_ids: 类型={type(instances.cluster_ids)}, 形状={instances.cluster_ids.shape}")
            if has_h_rel:
                logger.info(f"新样本 h_rel: 类型={type(instances.h_rel)}, 形状={instances.h_rel.shape}")
        else:
            logger.warning(f"新样本中没有instances字段!")
        
        # 应用TomatoFormat
        logger.info("开始应用TomatoFormat...")
        formatted = tomato_format(new_sample)
        logger.info("TomatoFormat应用完成")
        
        # 检查格式化后的样本
        logger.info(f"格式化后的样本字段: {list(formatted.keys())}")
        if "cluster_ids" in formatted:
            logger.info(f"格式化后的样本包含cluster_ids字段: 形状={formatted['cluster_ids'].shape}, 类型={formatted['cluster_ids'].dtype}")
            logger.info(f"cluster_ids内容: {formatted['cluster_ids']}")
        else:
            logger.warning("格式化后的样本不包含cluster_ids字段!")
            
        if "h_rel" in formatted:
            logger.info(f"格式化后的样本包含h_rel字段: 形状={formatted['h_rel'].shape}, 类型={formatted['h_rel'].dtype}")
            logger.info(f"h_rel内容: {formatted['h_rel']}")
        else:
            logger.warning("格式化后的样本不包含h_rel字段!")
        
        return True
    except Exception as e:
        logger.error(f"测试过程中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_augment_tomato()
    if success:
        logger.info("测试成功完成！")
    else:
        logger.error("测试失败！") 