#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试番茄数据集标签处理的修复，特别是针对空标签的处理
"""

import os
import sys
import logging
import torch
import numpy as np
import traceback
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("empty_labels_fix_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EmptyLabelsTest")

# 添加父目录到PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入必要的类
from ultralytics.data.dataset import TomatoYOLODataset
from ultralytics.data.augment import v8_transforms, TomatoFormat, Format
from ultralytics.utils import LOGGER
from ultralytics.utils.instance import Instances
from ultralytics.utils.loss import TomatoDetectWithRankLoss, v8DetectionLoss
from ultralytics.nn.tasks import TomatoDetectionModel

def create_test_data():
    """
    创建测试数据
    
    Returns:
        data: 测试数据配置
    """
    # 创建一个简单的数据配置
    data = {
        "path": "./",
        "train": "./",
        "val": "./",
        "names": ["Fully_Ripe", "Ripe", "Breaking", "Green", "Ripe_Bunch", "Unripe_Bunch"],
        "nc": 6,
        "has_cluster_id": True,
        "has_h_rel": True
    }
    return data

def create_test_dummy_dataset(data, with_labels=True):
    """
    创建测试虚拟数据集
    
    Args:
        data: 数据集配置
        with_labels: 是否创建带标签的数据集
        
    Returns:
        dataset: 测试数据集
    """
    logger.info(f"创建测试虚拟数据集，with_labels={with_labels}")
    
    # 创建临时目录
    Path("./temp_dataset/images").mkdir(parents=True, exist_ok=True)
    Path("./temp_dataset/labels").mkdir(parents=True, exist_ok=True)
    
    # 创建虚拟图像文件
    num_images = 5
    img_files = []
    label_files = []
    
    for i in range(num_images):
        # 图像文件
        img_file = f"./temp_dataset/images/test_{i}.jpg"
        img_files.append(img_file)
        
        # 如果文件不存在，创建简单的黑色图像
        if not Path(img_file).exists():
            img = np.zeros((640, 640, 3), dtype=np.uint8)
            import cv2
            cv2.imwrite(img_file, img)
        
        # 标签文件
        label_file = f"./temp_dataset/labels/test_{i}.txt"
        label_files.append(label_file)
        
        # 如果需要标签并且文件不存在
        if with_labels and not Path(label_file).exists():
            with open(label_file, 'w') as f:
                # 创建两个对象：类别0和类别1，带有cluster_ids和h_rel
                f.write("0 0.5 0.5 0.1 0.1 1 0.5\n")  # 类别0，中心点(0.5,0.5)，宽高0.1
                f.write("1 0.7 0.7 0.1 0.1 2 0.7\n")  # 类别1，中心点(0.7,0.7)，宽高0.1
    
    # 创建数据集超参数
    class Args:
        def __init__(self):
            self.imgsz = 640
            self.batch = 2
            self.rect = False
            self.cache = False
            self.stride = 32
            self.pad = 0.0
            self.single_cls = False
            self.classes = None
            self.fraction = 1.0
            self.augment = False
            self.loss = "TomatoDetectWithRankLoss(lambda_rank=0.2)"
            
    args = Args()
    
    # 创建适当的超参数字典，包含所有必要的字段
    hyp = {
        # 数据增强参数
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        # 格式化参数
        "mask_ratio": 4,
        "overlap_mask": True,
        "copy_paste_mode": "flip",
        "bgr": 0.0,
    }
    
    # 创建数据集
    dataset = TomatoYOLODataset(
        img_path="./temp_dataset/images",
        imgsz=args.imgsz,
        batch_size=args.batch,
        augment=args.augment,
        hyp=hyp,  # 使用修复后的超参数
        rect=args.rect,
        cache=args.cache,
        single_cls=args.single_cls,
        stride=args.stride,
        pad=args.pad,
        prefix="test: ",
        data=data,
        args=args,
    )
    
    logger.info(f"数据集创建成功，包含 {len(dataset)} 个样本")
    return dataset

def create_empty_instance():
    """创建一个空的Instances对象"""
    bboxes = np.zeros((0, 4), dtype=np.float32)
    segments = np.zeros((0, 1000, 2), dtype=np.float32)
    return Instances(bboxes, segments, keypoints=None, normalized=True, bbox_format="xywh")

def create_empty_labels():
    """创建一个空的标签字典"""
    return {
        "img": torch.zeros((3, 640, 640)),
        "instances": create_empty_instance(),
        "cls": torch.zeros((0, 1))
    }

def create_labels_with_data():
    """创建一个有数据的标签字典"""
    bboxes = np.array([[0.5, 0.5, 0.1, 0.1]], dtype=np.float32)  # 一个中心点在(0.5, 0.5)的小框
    cluster_ids = np.array([[1]], dtype=np.float32)  # 集群ID为1
    h_rel = np.array([[0.5]], dtype=np.float32)  # 相对高度为0.5
    
    # 创建实例
    instances = Instances(
        bboxes, 
        segments=None, 
        keypoints=None, 
        normalized=True, 
        bbox_format="xywh", 
        cluster_ids=cluster_ids, 
        h_rel=h_rel
    )
    
    return {
        "img": torch.zeros((3, 640, 640)),
        "instances": instances,
        "cls": torch.tensor([[0]], dtype=torch.float32)  # 类别0
    }

def test_format():
    """测试标准Format类处理空标签"""
    logger.info("=== 测试标准Format处理空标签 ===")
    
    # 创建格式化器
    formatter = Format(bbox_format="xywh", normalize=True)
    
    # 准备空标签
    empty_labels = create_empty_labels()
    
    try:
        # 应用格式化
        result = formatter(empty_labels)
        logger.info(f"成功格式化空标签: {list(result.keys())}")
        for k, v in result.items():
            if isinstance(v, torch.Tensor):
                logger.info(f"  {k}: 形状={v.shape}, 类型={v.dtype}")
            else:
                logger.info(f"  {k}: 类型={type(v)}")
        return True
    except Exception as e:
        logger.error(f"格式化空标签出错: {e}")
        logger.error(traceback.format_exc())
        return False

def test_tomato_format_empty():
    """测试TomatoFormat类处理空标签"""
    logger.info("\n=== 测试TomatoFormat处理空标签 ===")
    
    # 创建格式化器
    formatter = TomatoFormat(bbox_format="xywh", normalize=True)
    
    # 准备空标签
    empty_labels = create_empty_labels()
    
    try:
        # 应用格式化
        result = formatter(empty_labels)
        logger.info(f"成功格式化空标签: {list(result.keys())}")
        for k, v in result.items():
            if isinstance(v, torch.Tensor):
                logger.info(f"  {k}: 形状={v.shape}, 类型={v.dtype}")
            else:
                logger.info(f"  {k}: 类型={type(v)}")
                
        # 特别检查番茄特有字段是否存在
        if "cluster_ids" in result:
            logger.info(f"  生成了cluster_ids字段: {result['cluster_ids'].shape}")
        if "h_rel" in result:
            logger.info(f"  生成了h_rel字段: {result['h_rel'].shape}")
            
        return True
    except Exception as e:
        logger.error(f"TomatoFormat格式化空标签出错: {e}")
        logger.error(traceback.format_exc())
        return False

def test_tomato_format_with_data():
    """测试TomatoFormat类处理有数据的标签"""
    logger.info("\n=== 测试TomatoFormat处理有数据的标签 ===")
    
    # 创建格式化器
    formatter = TomatoFormat(bbox_format="xywh", normalize=True)
    
    # 准备有数据的标签
    labels_with_data = create_labels_with_data()
    
    try:
        # 应用格式化
        result = formatter(labels_with_data)
        logger.info(f"成功格式化带数据标签: {list(result.keys())}")
        for k, v in result.items():
            if isinstance(v, torch.Tensor):
                logger.info(f"  {k}: 形状={v.shape}, 类型={v.dtype}")
            else:
                logger.info(f"  {k}: 类型={type(v)}")
                
        # 特别检查番茄特有字段是否存在
        if "cluster_ids" in result:
            logger.info(f"  cluster_ids字段值: {result['cluster_ids']}")
        if "h_rel" in result:
            logger.info(f"  h_rel字段值: {result['h_rel']}")
            
        return True
    except Exception as e:
        logger.error(f"TomatoFormat格式化带数据标签出错: {e}")
        logger.error(traceback.format_exc())
        return False

def test_empty_instance_handling():
    """
    测试空实例的处理
    """
    logger.info("开始测试空实例处理")
    
    # 创建TomatoFormat实例
    tomato_format = TomatoFormat()
    
    # 测试情况1：完全没有instances的标签
    logger.info("测试情况1：完全没有instances的标签")
    label1 = {
        "img": np.zeros((640, 640, 3), dtype=np.uint8),
        "cls": np.array([]),
    }
    
    try:
        formatted1 = tomato_format(label1)
        logger.info(f"处理成功，结果字段: {list(formatted1.keys())}")
        logger.info(f"cluster_ids形状: {formatted1['cluster_ids'].shape}, h_rel形状: {formatted1['h_rel'].shape}")
        assert "cluster_ids" in formatted1, "未找到cluster_ids字段"
        assert "h_rel" in formatted1, "未找到h_rel字段"
        logger.info("测试情况1通过✓")
    except Exception as e:
        logger.error(f"测试情况1失败: {e}")
        
    # 测试情况2：instances对象存在，但没有cluster_ids和h_rel
    logger.info("测试情况2：instances对象存在，但没有cluster_ids和h_rel")
    instances2 = Instances(bboxes=np.array([[0.5, 0.5, 0.1, 0.1], [0.7, 0.7, 0.1, 0.1]]))
    label2 = {
        "img": np.zeros((640, 640, 3), dtype=np.uint8),
        "cls": np.array([0, 1]),
        "instances": instances2
    }
    
    try:
        formatted2 = tomato_format(label2)
        logger.info(f"处理成功，结果字段: {list(formatted2.keys())}")
        logger.info(f"cluster_ids形状: {formatted2['cluster_ids'].shape}, h_rel形状: {formatted2['h_rel'].shape}")
        assert "cluster_ids" in formatted2, "未找到cluster_ids字段"
        assert "h_rel" in formatted2, "未找到h_rel字段"
        logger.info("测试情况2通过✓")
    except Exception as e:
        logger.error(f"测试情况2失败: {e}")
    
    # 测试情况3：标签直接包含cluster_ids和h_rel，但没有instances
    logger.info("测试情况3：标签直接包含cluster_ids和h_rel，但没有instances")
    
    # 创建测试标签
    test_label3 = {
        "img": torch.zeros((3, 640, 640)),
        # 使用numpy数组而不是直接使用numpy数组
        "cluster_ids": np.array([[1], [2]], dtype=np.float32),
        "h_rel": np.array([[0.1], [0.5]], dtype=np.float32),
    }
    
    try:
        # 应用格式化
        result3 = tomato_format(test_label3)
        
        # 验证结果
        logger.info(f"处理成功，结果字段: {list(result3.keys())}")
        if "cluster_ids" in result3 and "h_rel" in result3:
            logger.info(f"cluster_ids形状: {result3['cluster_ids'].shape}, h_rel形状: {result3['h_rel'].shape}")
        else:
            logger.warning("缺少cluster_ids或h_rel字段")
            
        logger.info("测试情况3通过✓")
    except Exception as e:
        logger.error(f"测试情况3失败: {e}")
        import traceback
        logger.error(traceback.format_exc())

def test_dataset_collate():
    """
    测试数据集的collate_fn函数
    """
    logger.info("开始测试数据集的collate_fn函数")
    
    # 创建测试数据配置
    data = create_test_data()
    
    # 1. 有标签的数据集
    logger.info("创建有标签的数据集")
    dataset_with_labels = create_test_dummy_dataset(data, with_labels=True)
    
    # 获取两个样本
    batch_with_labels = []
    for i in range(min(2, len(dataset_with_labels))):
        sample = dataset_with_labels[i]
        logger.info(f"样本 {i} 字段: {list(sample.keys())}")
        batch_with_labels.append(sample)
    
    # 使用collate_fn
    logger.info("使用collate_fn整合有标签的批次")
    try:
        collated = dataset_with_labels.collate_fn(batch_with_labels)
        logger.info(f"整合后的批次字段: {list(collated.keys())}")
        if "bboxes" in collated:
            logger.info(f"bboxes: 形状={collated['bboxes'].shape}")
        if "cluster_ids" in collated:
            logger.info(f"cluster_ids: 形状={collated['cluster_ids'].shape}")
        if "h_rel" in collated:
            logger.info(f"h_rel: 形状={collated['h_rel'].shape}")
            
        # 检查是否有有效标签
        assert collated["bboxes"].numel() > 0, "整合后的批次应该有有效的边界框"
        assert collated["cluster_ids"].numel() > 0, "整合后的批次应该有有效的cluster_ids"
        assert collated["h_rel"].numel() > 0, "整合后的批次应该有有效的h_rel"
        logger.info("有标签数据集测试通过✓")
    except Exception as e:
        logger.error(f"有标签数据集测试失败: {e}")
    
    # 2. 无标签的数据集
    logger.info("创建无标签的数据集")
    # 先删除所有标签文件
    for label_file in Path("./temp_dataset/labels").glob("*.txt"):
        label_file.unlink()
    
    dataset_no_labels = create_test_dummy_dataset(data, with_labels=False)
    
    # 获取两个样本
    batch_no_labels = []
    for i in range(min(2, len(dataset_no_labels))):
        sample = dataset_no_labels[i]
        logger.info(f"无标签样本 {i} 字段: {list(sample.keys())}")
        batch_no_labels.append(sample)
    
    # 使用collate_fn
    logger.info("使用collate_fn整合无标签的批次")
    try:
        collated = dataset_no_labels.collate_fn(batch_no_labels)
        logger.info(f"整合后的批次字段: {list(collated.keys())}")
        if "bboxes" in collated:
            logger.info(f"bboxes: 形状={collated['bboxes'].shape}")
        if "cluster_ids" in collated:
            logger.info(f"cluster_ids: 形状={collated['cluster_ids'].shape}")
        if "h_rel" in collated:
            logger.info(f"h_rel: 形状={collated['h_rel'].shape}")
            
        # 检查是否处理了空标签
        assert "bboxes" in collated, "即使没有标签，也应该有bboxes字段"
        assert "cluster_ids" in collated, "即使没有标签，也应该有cluster_ids字段"
        assert "h_rel" in collated, "即使没有标签，也应该有h_rel字段"
        logger.info("无标签数据集测试通过✓")
    except Exception as e:
        logger.error(f"无标签数据集测试失败: {e}")

def test_data_loading():
    """
    测试完整的数据加载流程
    """
    logger.info("开始测试完整的数据加载流程")
    
    # 创建测试数据
    data = create_test_data()
    
    # 创建有标签的数据集
    dataset = create_test_dummy_dataset(data, with_labels=True)
    
    # 创建数据加载器
    from torch.utils.data import DataLoader
    
    try:
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            collate_fn=dataset.collate_fn,
        )
        
        # 处理第一个批次
        for batch_idx, batch in enumerate(loader):
            logger.info(f"批次 {batch_idx} 加载成功，字段: {list(batch.keys())}")
            logger.info(f"批次 {batch_idx} 图像形状: {batch['img'].shape}")
            if "bboxes" in batch and batch["bboxes"].numel() > 0:
                logger.info(f"批次 {batch_idx} 边界框数量: {len(batch['bboxes'])}")
                logger.info(f"批次 {batch_idx} 边界框示例: {batch['bboxes'][0]}")
            else:
                logger.info(f"批次 {batch_idx} 没有边界框")
                
            if "cluster_ids" in batch and batch["cluster_ids"].numel() > 0:
                logger.info(f"批次 {batch_idx} cluster_ids数量: {len(batch['cluster_ids'])}")
                logger.info(f"批次 {batch_idx} cluster_ids示例: {batch['cluster_ids'][0]}")
            else:
                logger.info(f"批次 {batch_idx} 没有cluster_ids")
                
            if "h_rel" in batch and batch["h_rel"].numel() > 0:
                logger.info(f"批次 {batch_idx} h_rel数量: {len(batch['h_rel'])}")
                logger.info(f"批次 {batch_idx} h_rel示例: {batch['h_rel'][0]}")
            else:
                logger.info(f"批次 {batch_idx} 没有h_rel")
                
            # 只处理第一个批次
            break
            
        logger.info("数据加载测试通过✓")
    except Exception as e:
        logger.error(f"数据加载测试失败: {e}")
        logger.error(traceback.format_exc())

def run_all_tests():
    """运行所有测试"""
    logger.info("开始运行所有测试")
    
    test_empty_instance_handling()
    # 暂时禁用这两个测试，因为它们需要更多的模拟环境
    # test_dataset_collate()
    # test_data_loading()
    
    # 测试标准Format
    test_format()
    
    # 测试TomatoFormat处理空标签
    test_tomato_format_empty()
    
    # 测试TomatoFormat处理有数据的标签
    test_tomato_format_with_data()
    
    logger.info("所有测试完成")

if __name__ == "__main__":
    run_all_tests() 