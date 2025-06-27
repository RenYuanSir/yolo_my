#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试番茄检测模型的训练过程，验证修复是否在实际训练中有效
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
import logging
import traceback
from copy import deepcopy
from types import SimpleNamespace

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tomato_training_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_training")

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 应用补丁
try:
    from patch_loss import apply_patches
    apply_patches()
    print("patches: 成功应用补丁")
except Exception as e:
    print(f"patches: 应用补丁时出错: {e}")

# 导入必要的类
from ultralytics.data.dataset import TomatoYOLODataset
from ultralytics.data.augment import v8_transforms, TomatoFormat
from ultralytics.utils import LOGGER
from ultralytics.utils.instance import Instances
from ultralytics.utils.loss import TomatoDetectWithRankLoss, v8DetectionLoss
from ultralytics.nn.tasks import TomatoDetectionModel
from ultralytics.utils.tal import make_anchors  # 导入make_anchors函数
from ultralytics.utils.metrics import bbox_iou  # 导入bbox_iou函数

# 从test_augment_tomato.py导入模拟数据集
from test_augment_tomato import MockTomatoDataset, create_mock_data

# 创建一个简单的批次收集器函数
def collate_fn(batch):
    """将数据样本整合为批次"""
    try:
        logger.info(f"开始整合批次，批次大小: {len(batch)}")
        new_batch = {}
        keys = batch[0].keys()
        logger.info(f"样本字段: {keys}")
        
        values = list(zip(*[list(b.values()) for b in batch]))
        
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
                logger.info(f"整合后的图像形状: {value.shape}")
            elif k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
                logger.info(f"整合后的{k}形状: {value.shape}")
            elif k in {"cluster_ids", "h_rel"}:
                # 确保所有元素都是PyTorch张量
                tensors = []
                for v in value:
                    if isinstance(v, np.ndarray):
                        tensors.append(torch.from_numpy(v))
                    else:
                        tensors.append(v)
                value = torch.cat(tensors, 0)
                logger.info(f"整合后的{k}形状: {value.shape}")
            elif k == "instances":
                # 特殊处理instances
                logger.info(f"整合instances，数量: {len(value)}")
                value = value  # 不做特殊处理，保持列表形式
            else:
                logger.info(f"保留原始{k}")
            new_batch[k] = value
            
        # 处理批次索引
        if "batch_idx" in new_batch:
            new_batch["batch_idx"] = list(new_batch["batch_idx"])
            for i in range(len(new_batch["batch_idx"])):
                new_batch["batch_idx"][i] += i  # 添加目标图像索引
            new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
            logger.info(f"整合后的batch_idx形状: {new_batch['batch_idx'].shape}")
        
        logger.info("批次整合成功")
        return new_batch
    except Exception as e:
        logger.error(f"整合批次时出错: {e}")
        logger.error(traceback.format_exc())
        raise

def create_mock_model():
    """创建一个模拟的番茄检测模型"""
    try:
        logger.info("开始创建模拟模型...")
        # 使用相对路径，确保找到正确的模型配置文件
        model_yaml = "ultralytics/cfg/models/v12/yolo12-A2C2f-DYT-EfficientHead-MambaOut.yaml"
        logger.info(f"使用模型配置: {model_yaml}")
        
        model = TomatoDetectionModel(model_yaml, nc=6)
        logger.info("模型创建成功")
        
        # 添加args属性，这是必要的，但不修改原生YOLO文件
        model.args = {
            "box": 7.5,  # box loss gain
            "cls": 0.5,  # cls loss gain
            "dfl": 1.5,  # dfl loss gain
            "pose": 12.0,  # pose loss gain
            "kobj": 1.0,  # keypoint obj loss gain
            "fl_gamma": 1.5,  # focal loss gamma
            "label_smoothing": 0.0,  # label smoothing
            "nbs": 64,  # nominal batch size
            "hsv_h": 0.015,  # image HSV-Hue augmentation (fraction)
            "hsv_s": 0.7,  # image HSV-Saturation augmentation (fraction)
            "hsv_v": 0.4,  # image HSV-Value augmentation (fraction)
            "degrees": 0.0,  # image rotation (+/- deg)
            "translate": 0.1,  # image translation (+/- fraction)
            "scale": 0.5,  # image scale (+/- gain)
            "shear": 0.0,  # image shear (+/- deg)
            "perspective": 0.0,  # image perspective (+/- fraction), range 0-0.001
            "flipud": 0.0,  # image flip up-down (probability)
            "fliplr": 0.5,  # image flip left-right (probability)
            "mosaic": 1.0,  # image mosaic (probability)
            "mixup": 0.0,  # image mixup (probability)
            "copy_paste": 0.0,  # segment copy-paste (probability)
            "auto_augment": "randaugment",  # auto augmentation policy for classification (randaugment, autoaugment, augmix)
            "erasing": 0.4,  # probability of random erasing during classification training
            "crop_fraction": 1.0,  # image crop fraction for classification (default=1.0, no cropping)
        }
        
        return model
    except Exception as e:
        logger.error(f"创建模型时出错: {e}")
        logger.error(traceback.format_exc())
        raise

def debug_bboxes(name, bboxes):
    """打印边界框信息以便调试"""
    if not isinstance(bboxes, torch.Tensor):
        print(f"{name}: 不是张量")
        return
        
    print(f"{name}形状: {bboxes.shape}")
    if bboxes.numel() > 0:
        print(f"{name}范围: [{bboxes.min().item():.4f}, {bboxes.max().item():.4f}]")
        if bboxes.shape[-1] == 4:
            # 假设是XYWH或XYXY格式
            if torch.all(bboxes == 0):
                print(f"{name}所有元素都是0!")
            else:
                first_box = bboxes[0] if bboxes.dim() == 2 else bboxes[0, 0]
                print(f"{name}第一个框: [{first_box[0]:.4f}, {first_box[1]:.4f}, {first_box[2]:.4f}, {first_box[3]:.4f}]")
    else:
        print(f"{name}为空")

def test_training():
    """测试番茄检测模型的训练过程"""
    logger.info("开始测试番茄检测模型的训练过程")
    
    try:
        # 创建模拟数据集
        dataset = MockTomatoDataset(num_samples=10)
        logger.info(f"创建模拟数据集成功: {len(dataset)} 张图像")
        
        # 创建参数
        class Args:
            def __init__(self):
                self.imgsz = 640
                self.batch = 2
                self.rect = False
                self.cache = None
                self.single_cls = False
                self.augment = False  # 关闭增强，简化测试
                self.stride = 32
                self.pad = 0.0
                self.loss = "TomatoDetectWithRankLoss"  # 确保启用cluster_ids和h_rel
                # 添加必要的超参数
                self.mosaic = 0.0
                self.mixup = 0.0
                self.copy_paste = 0.0
                self.copy_paste_mode = "flip"  # 添加缺少的属性
                self.mask_ratio = 4.0
                self.overlap_mask = True
                self.degrees = 0.0
                self.translate = 0.0
                self.scale = 0.0
                self.shear = 0.0
                self.perspective = 0.0
                self.flipud = 0.0
                self.fliplr = 0.0
                self.hsv_h = 0.0
                self.hsv_s = 0.0
                self.hsv_v = 0.0
                # 添加损失函数需要的超参数
                self.box = 7.5  # box loss gain
                self.cls = 0.5  # cls loss gain
                self.dfl = 1.5  # dfl loss gain
                self.fl_gamma = 1.5  # focal loss gamma
                self.label_smoothing = 0.0  # label smoothing
                self.nbs = 64  # nominal batch size
        
        args = Args()
        logger.info("创建参数成功")
        
        # 创建数据变换
        logger.info("开始创建数据变换...")
        transforms = v8_transforms(dataset, args.imgsz, args)
        logger.info("创建数据变换成功")
        
        # 手动创建数据集
        logger.info("开始准备样本...")
        samples = []
        for i in range(5):
            sample = dataset.get_image_and_label(i)
            logger.info(f"获取样本 {i} 成功")
            
            # 检查原始样本中是否包含必要的字段
            if "instances" in sample:
                instances = sample["instances"]
                logger.info(f"原始样本 {i} 包含instances字段，类型={type(instances)}")
                
                has_cluster_ids = hasattr(instances, 'cluster_ids') and instances.cluster_ids is not None
                has_h_rel = hasattr(instances, 'h_rel') and instances.h_rel is not None
                
                if has_cluster_ids:
                    logger.info(f"原始样本 {i} 的instances包含cluster_ids字段，形状={instances.cluster_ids.shape}")
                else:
                    logger.warning(f"原始样本 {i} 的instances不包含cluster_ids字段!")
                
                if has_h_rel:
                    logger.info(f"原始样本 {i} 的instances包含h_rel字段，形状={instances.h_rel.shape}")
                else:
                    logger.warning(f"原始样本 {i} 的instances不包含h_rel字段!")
            else:
                logger.warning(f"原始样本 {i} 不包含instances字段!")
            
            transformed_sample = transforms(sample)
            logger.info(f"变换样本 {i} 成功")
            
            # 检查变换后的样本中是否包含必要的字段
            if "cluster_ids" in transformed_sample:
                logger.info(f"变换后样本 {i} 包含cluster_ids字段，形状={transformed_sample['cluster_ids'].shape}")
            else:
                logger.warning(f"变换后样本 {i} 不包含cluster_ids字段!")
                
            if "h_rel" in transformed_sample:
                logger.info(f"变换后样本 {i} 包含h_rel字段，形状={transformed_sample['h_rel'].shape}")
            else:
                logger.warning(f"变换后样本 {i} 不包含h_rel字段!")
            
            samples.append(transformed_sample)
        logger.info(f"准备了 {len(samples)} 个样本")
        
        # 检查变换后的样本
        logger.info("检查变换后的第一个样本...")
        sample0 = samples[0]
        logger.info(f"样本字段: {list(sample0.keys())}")
        
        if "instances" in sample0:
            logger.info(f"样本包含instances字段，类型={type(sample0['instances'])}")
            instances = sample0["instances"]
            if hasattr(instances, 'cluster_ids') and instances.cluster_ids is not None:
                logger.info(f"instances.cluster_ids存在，形状={instances.cluster_ids.shape}")
            else:
                logger.warning("instances.cluster_ids不存在!")
            if hasattr(instances, 'h_rel') and instances.h_rel is not None:
                logger.info(f"instances.h_rel存在，形状={instances.h_rel.shape}")
            else:
                logger.warning("instances.h_rel不存在!")
        else:
            logger.warning("样本不包含instances字段!")
            
        if "cluster_ids" in sample0:
            logger.info(f"样本包含独立的cluster_ids字段，形状={sample0['cluster_ids'].shape}")
        else:
            logger.warning("样本不包含独立的cluster_ids字段!")
            
        if "h_rel" in sample0:
            logger.info(f"样本包含独立的h_rel字段，形状={sample0['h_rel'].shape}")
        else:
            logger.warning("样本不包含独立的h_rel字段!")
        
        # 创建批次
        logger.info("开始创建批次...")
        batch = collate_fn(samples)
        logger.info("创建批次成功")
        logger.info(f"批次字段: {list(batch.keys())}")
        
        # 检查批次中的关键字段
        if "cluster_ids" in batch:
            logger.info(f"批次包含cluster_ids字段，形状={batch['cluster_ids'].shape}，类型={batch['cluster_ids'].dtype}")
            logger.info(f"cluster_ids样本: {batch['cluster_ids'][:5]}")
        else:
            logger.warning("批次不包含cluster_ids字段!")
            
        if "h_rel" in batch:
            logger.info(f"批次包含h_rel字段，形状={batch['h_rel'].shape}，类型={batch['h_rel'].dtype}")
            logger.info(f"h_rel样本: {batch['h_rel'][:5]}")
        else:
            logger.warning("批次不包含h_rel字段!")
            
        if "instances" in batch:
            logger.info(f"批次包含instances字段，类型={type(batch['instances'])}")
        else:
            logger.warning("批次不包含instances字段!")
        
        # 调试边界框数据
        debug_bboxes("原始边界框", batch["bboxes"])
        
        # 创建模型和损失函数
        logger.info("开始创建模型和损失函数...")
        device = torch.device("cpu")
        model = create_mock_model()
        model.to(device)
        logger.info("创建模型成功")
        
        # 修改v8DetectionLoss的初始化方法，使其不依赖model.args
        logger.info("修改v8DetectionLoss初始化方法...")
        
        # 保存原始的__init__方法
        original_init = v8DetectionLoss.__init__
        
        # 定义新的__init__方法
        def new_init(self, model, tal_topk=10):
            """临时替换的初始化方法，不依赖model.args"""
            device = next(model.parameters()).device  # get model device
            
            # 使用SimpleNamespace替代字典，以支持属性访问
            h = SimpleNamespace(
                box=7.5,  # box loss gain
                cls=0.5,  # cls loss gain
                dfl=1.5,  # dfl loss gain
                fl_gamma=1.5,  # focal loss gamma
                label_smoothing=0.0,  # label smoothing
                nbs=64,  # nominal batch size
            )
            
            # 获取模型的最后一层
            m = model.model[-1]  # Detect() module
            self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")
            self.hyp = h
            self.stride = m.stride  # model strides
            self.nc = m.nc  # number of classes
            self.no = m.nc + m.reg_max * 4
            self.reg_max = m.reg_max
            self.device = device

            self.use_dfl = m.reg_max > 1

            from ultralytics.utils.tal import TaskAlignedAssigner
            from ultralytics.utils.loss import BboxLoss
            
            self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
            self.bbox_loss = BboxLoss(m.reg_max).to(device)
            self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        
        # 临时替换__init__方法
        v8DetectionLoss.__init__ = new_init
        logger.info("v8DetectionLoss初始化方法修改成功")
        
        # 创建损失函数
        logger.info("创建损失函数...")
        try:
            loss_fn = TomatoDetectWithRankLoss(
                model=model,
                lambda_rank=0.2,
                margin=0.05,
                tal_topk=10
            )
            logger.info("损失函数创建成功")
        except Exception as e:
            logger.error(f"创建损失函数时出错: {e}")
            logger.error(traceback.format_exc())
            return False
        finally:
            # 恢复原始的__init__方法
            v8DetectionLoss.__init__ = original_init
            logger.info("恢复v8DetectionLoss初始化方法")
        
        # 将批次移到设备上
        logger.info("开始将批次移到设备上...")
        batch_on_device = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_on_device[k] = v.to(device)
                logger.info(f"将{k}移到设备上，形状={v.shape}，类型={v.dtype}")
            else:
                batch_on_device[k] = v
                logger.info(f"保留{k}，类型={type(v)}")
        logger.info("批次已移到设备上")
        
        # 确保必要的字段存在
        for field in ["cls", "bboxes", "cluster_ids", "h_rel"]:
            if field not in batch_on_device:
                logger.warning(f"批次中缺少{field}字段！排序损失可能无法计算")
        
        # 前向传播
        logger.info("开始前向传播...")
        try:
            # 确保图像是正确的格式
            img = batch_on_device["img"]
            logger.info(f"输入图像类型: {img.dtype}, 形状: {img.shape}, 范围: [{img.min()}, {img.max()}]")
            
            # 确保图像是浮点类型
            if img.dtype != torch.float32:
                img = img.float()
                # 如果图像范围是0-255，则归一化到0-1
                if img.max() > 1.0:
                    img = img / 255.0
                logger.info(f"转换后的图像类型: {img.dtype}, 范围: [{img.min()}, {img.max()}]")
                batch_on_device["img"] = img
            
            # 尝试直接使用模型进行前向传播
            logger.info("尝试直接进行前向传播...")
            try:
                # 确保模型处于训练模式
                model.train()
                preds = model(img)
                logger.info("直接前向传播成功")
            except Exception as e:
                logger.error(f"直接前向传播出错: {e}")
                logger.error(traceback.format_exc())
                
                # 尝试使用预处理方法
                logger.info("尝试使用预处理方法...")
                try:
                    # 确保图像尺寸是模型期望的尺寸
                    if img.shape[-2:] != (640, 640):
                        # 使用插值调整大小
                        img_resized = torch.nn.functional.interpolate(
                            img, size=(640, 640), mode='bilinear', align_corners=False
                        )
                        logger.info(f"调整图像大小: {img.shape} -> {img_resized.shape}")
                        img = img_resized
                    
                    # 再次尝试前向传播
                    preds = model(img)
                    logger.info("预处理后前向传播成功")
                except Exception as e:
                    logger.error(f"预处理后前向传播出错: {e}")
                    logger.error(traceback.format_exc())
                    return False
            
            logger.info("前向传播成功")
            logger.info(f"预测结果类型: {type(preds)}")
        except Exception as e:
            logger.error(f"前向传播时出错: {e}")
            logger.error(traceback.format_exc())
            return False
        
        # 计算损失
        logger.info("开始计算损失...")
        logger.info(f"预测结果是字典，包含键: {list(preds.keys())}")
        logger.info("检查批次数据:")
        for k, v in batch_on_device.items():
            logger.info(f"  {k}: 类型={type(v) if not isinstance(v, torch.Tensor) else v.dtype}, 形状={v.shape if isinstance(v, torch.Tensor) else None}")
        
        # 调试边界框数据
        debug_bboxes("原始边界框", batch_on_device["bboxes"])
        
        # 手动计算排序损失，检查中间结果
        logger.info("手动计算排序损失，检查中间结果...")
        rank_loss = loss_fn.compute_ranking_loss(batch_on_device)
        logger.info(f"手动计算的排序损失: {rank_loss.item()}")
        
        # 深入检查损失函数输入
        logger.info("深入检查损失函数输入...")
        
        # 跟踪预处理前后的边界框
        original_preprocess = v8DetectionLoss.preprocess
        
        def tracking_preprocess(self, targets, batch_size, scale_tensor):
            print(f"预处理前targets形状: {targets.shape}")
            print(f"预处理前targets边界框: {targets[:5, 1:5]}")  # 打印前5个样本
            result = original_preprocess(self, targets, batch_size, scale_tensor)
            print(f"预处理后result形状: {result.shape}")
            print(f"预处理后result边界框: {result[0, :5, 1:5]}")  # 打印第一个批次的前5个样本
            return result
        
        # 替换preprocess方法进行调试
        v8DetectionLoss.preprocess = tracking_preprocess
        
        try:
            loss = loss_fn(preds, batch_on_device)
        finally:
            # 恢复原始方法
            v8DetectionLoss.preprocess = original_preprocess
        
        # 反向传播
        logger.info("开始反向传播...")
        try:
            # 处理返回的损失
            if isinstance(loss, tuple):
                # 损失是一个元组（total_loss, loss_items）
                total_loss, loss_items = loss
                logger.info(f"损失计算成功，总损失: {total_loss.item()}, 损失项: {loss_items}")
                # 打印每个损失分量
                logger.info(f"边界框损失: {loss_items[0].item()}")
                logger.info(f"分类损失: {loss_items[1].item()}")
                logger.info(f"DFL损失: {loss_items[2].item()}")
                logger.info(f"排序损失: {loss_items[3].item() if len(loss_items) > 3 else 0.0}")
                
                # 对总损失进行反向传播
                if isinstance(total_loss, torch.Tensor):
                    total_loss.backward()
                    logger.info("反向传播成功")
                else:
                    logger.error(f"总损失不是张量，无法进行反向传播: {type(total_loss)}")
                    return False
            else:
                # 损失直接是一个张量
                if isinstance(loss, torch.Tensor):
                    loss.backward()
                    logger.info("反向传播成功")
                else:
                    logger.error(f"损失不是张量，无法进行反向传播: {type(loss)}")
                    return False
        except Exception as e:
            logger.error(f"反向传播过程中出错: {e}")
            logger.error(traceback.format_exc())
            return False
        
        return True
    except Exception as e:
        logger.error(f"测试过程中出错: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    try:
        success = test_training()
        if success:
            logger.info("测试成功完成！")
        else:
            logger.error("测试失败！")
    except Exception as e:
        logger.error(f"测试主函数出错: {e}")
        logger.error(traceback.format_exc()) 