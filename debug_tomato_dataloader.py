#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import yaml
import logging
from pathlib import Path


# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('debug_tomato_dataloader')

# 添加YOLO路径
YOLO_PATH = os.path.dirname(os.path.abspath(__file__))
print(f"YOLO_PATH: {YOLO_PATH}")
print(f"YOLO_PATH存在: {os.path.exists(YOLO_PATH)}")
if not os.path.exists(YOLO_PATH):
    logger.error(f"YOLO路径不存在: {YOLO_PATH}")
    sys.exit(1)
sys.path.append(YOLO_PATH)

# 导入YOLO相关模块
try:
    # 确保本地路径优先
    sys.path.insert(0, YOLO_PATH)
    
    # 从ultralytics模块导入
    from ultralytics.data.dataset import TomatoYOLODataset
    from ultralytics.utils import LOGGER, colorstr
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import TomatoDetectionModel
    from ultralytics.models.yolo.tomato.train import TomatoTrainer
    from ultralytics.utils.loss import TomatoDetectWithRankLoss
    logger.info("成功导入YOLO模块")
except ImportError as e:
    logger.error(f"导入YOLO模块失败: {e}")
    sys.exit(1)

def visualize_batch(batch, save_dir="debug_outputs", max_images=4):
    """可视化批次数据"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取图像和标签
    images = batch["img"]
    bboxes = batch["bboxes"]
    cls_labels = batch["cls"]
    batch_idx = batch["batch_idx"]
    
    # 获取番茄特有属性
    cluster_ids = batch.get("cluster_ids", None)
    h_rel = batch.get("h_rel", None)
    
    # 确定要显示的图像数量
    num_images = min(max_images, images.shape[0])
    
    # 记录批次信息
    logger.info(f"批次信息:")
    logger.info(f"- 图像数量: {images.shape[0]}")
    logger.info(f"- 图像形状: {images.shape}")
    logger.info(f"- 边界框数量: {bboxes.shape[0]}")
    logger.info(f"- 类别标签数量: {cls_labels.shape[0]}")
    
    if cluster_ids is not None:
        logger.info(f"- 串ID数量: {cluster_ids.shape[0]}")
        logger.info(f"- 串ID样本: {cluster_ids[:5].tolist()}")
    
    if h_rel is not None:
        logger.info(f"- 相对高度数量: {h_rel.shape[0]}")
        logger.info(f"- 相对高度样本: {h_rel[:5].tolist()}")
    
    # 检测边界框格式
    # 检查第三和第四列的值是否小于第一和第二列
    # 如果大多数情况是这样，那么可能是xyxy格式
    w_check = (bboxes[:, 2] < bboxes[:, 0]).float().mean() < 0.1
    h_check = (bboxes[:, 3] < bboxes[:, 1]).float().mean() < 0.1

    # 检查所有值是否都在0-1之间（归一化格式）
    normalized = (bboxes >= 0).all() and (bboxes <= 1).all()

    # 检查第三和第四列的值是否很小（可能是中心点+宽高格式）
    small_wh = (bboxes[:, 2] < 0.5).float().mean() > 0.9 and (bboxes[:, 3] < 0.5).float().mean() > 0.9

    # 判断格式
    if not w_check and not h_check and normalized:
        # 如果第三和第四列小于第一和第二列，可能是中心点+宽高
        bbox_format = "xywh_center"
        logger.info(f"- 检测到的边界框格式: {bbox_format} (中心点坐标+宽高)")
    elif w_check and h_check and normalized:
        # 如果满足正常的宽高约束，可能是xyxy格式
        bbox_format = "xyxy"
        logger.info(f"- 检测到的边界框格式: {bbox_format} (左上右下坐标)")
    else:
        # 打印更多信息以进行调试
        bbox_format = "unknown"
        logger.info(f"- 无法确定边界框格式，假设为xywh_center格式")
        logger.info(f"- w_check: {w_check}, h_check: {h_check}, normalized: {normalized}, small_wh: {small_wh}")
        # 查看几个样本
        for i in range(min(5, len(bboxes))):
            b = bboxes[i].tolist()
            logger.info(f"- 边界框样本[{i}]: {b}, w={b[2]-b[0]:.4f}, h={b[3]-b[1]:.4f}")
        
        bbox_format = "xywh_center"  # 默认使用这个格式处理
    
    # 遍历图像进行可视化
    for i in range(num_images):
        # 获取当前图像
        img = images[i].permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
        
        # 标准化图像用于显示
        img = (img * 255).astype(np.uint8)
        
        # 创建图形
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        
        # 获取当前图像的边界框
        img_mask = batch_idx == i
        img_bboxes = bboxes[img_mask]
        img_cls = cls_labels[img_mask]
        
        # 获取当前图像的番茄特有属性
        img_cluster_ids = None
        img_h_rel = None
        if cluster_ids is not None:
            img_cluster_ids = cluster_ids[img_mask]
        if h_rel is not None:
            img_h_rel = h_rel[img_mask]
        
        # 绘制边界框
        for j, bbox in enumerate(img_bboxes):
            # 根据检测到的格式处理边界框
            if bbox_format == "xyxy":
                x1, y1, x2, y2 = bbox.tolist()
                x1 *= img.shape[1]  # 缩放到图像尺寸
                x2 *= img.shape[1]
                y1 *= img.shape[0]
                y2 *= img.shape[0]
            elif bbox_format == "xywh_center":
                x, y, w, h = bbox.tolist()
                # 转换为像素坐标
                x *= img.shape[1]
                y *= img.shape[0]
                w *= img.shape[1]
                h *= img.shape[0]
                # 计算左上角和右下角
                x1 = x - w/2
                y1 = y - h/2
                x2 = x + w/2
                y2 = y + h/2
            else:
                # 不确定的格式，尝试使用默认的处理方式
                # 假设是xyxy格式但值域可能不同
                x1, y1, x2, y2 = bbox.tolist()
                
                # 检查是否需要缩放（如果值都很小，可能是归一化坐标）
                if max(x1, y1, x2, y2) < 10:  # 假设是归一化坐标
                    x1 *= img.shape[1]
                    x2 *= img.shape[1]
                    y1 *= img.shape[0]
                    y2 *= img.shape[0]
            
            # 获取类别标签
            cls_id = int(img_cls[j].item())
            
            # 获取番茄特有属性
            cluster_id = -1
            h_rel_val = -1
            if img_cluster_ids is not None:
                cluster_id = img_cluster_ids[j].item()
            if img_h_rel is not None:
                h_rel_val = img_h_rel[j].item()
            
            # 绘制边界框
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(rect)
            
            # 添加标签
            label = f"C{cls_id}"
            if cluster_id >= 0:
                label += f", ID{int(cluster_id)}"
            if h_rel_val >= 0:
                label += f", H{h_rel_val:.2f}"
            
            plt.text(x1, y1-5, label, color='white', fontsize=10, 
                     bbox=dict(facecolor='red', alpha=0.5))
        
        # 保存图像
        plt.title(f"Image {i+1}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"debug_image_{i+1}.png"))
        plt.close()
        
        logger.info(f"已保存图像 {i+1} 的可视化结果")
    
    return os.path.join(save_dir, f"debug_image_1.png")

def check_model_input(model, batch):
    """检查模型输入"""
    logger.info("检查模型输入...")
    
    # 提取必要的输入
    images = batch["img"]
    
    # 记录输入形状
    logger.info(f"模型输入形状: {images.shape}")
    
    # 检查模型是否有forward方法
    if not hasattr(model, 'forward'):
        logger.error("模型没有forward方法")
        return False
    
    # 尝试进行前向传播
    try:
        with torch.no_grad():
            # 仅使用图像进行前向传播
            outputs = model(images)
            logger.info(f"模型前向传播成功，输出类型: {type(outputs)}")
            
            # 检查输出结构
            if isinstance(outputs, tuple):
                logger.info(f"输出是元组，长度: {len(outputs)}")
                for i, out in enumerate(outputs):
                    if isinstance(out, torch.Tensor):
                        logger.info(f"- 输出[{i}] 形状: {out.shape}")
                    elif isinstance(out, dict):
                        logger.info(f"- 输出[{i}] 是字典，键: {list(out.keys())}")
                    else:
                        logger.info(f"- 输出[{i}] 类型: {type(out)}")
            elif isinstance(outputs, torch.Tensor):
                logger.info(f"输出是张量，形状: {outputs.shape}")
            elif isinstance(outputs, dict):
                logger.info(f"输出是字典，键: {list(outputs.keys())}")
                for k, v in outputs.items():
                    if isinstance(v, torch.Tensor):
                        logger.info(f"- {k}: 形状={v.shape}")
                    else:
                        logger.info(f"- {k}: 类型={type(v)}")
            else:
                logger.info(f"输出类型: {type(outputs)}")
            
            return True
    except Exception as e:
        logger.error(f"模型前向传播失败: {e}")
        return False

def check_loss_calculation(model, batch):
    """检查损失计算"""
    logger.info("检查损失计算...")
    
    # 创建损失函数
    try:
        criterion = TomatoDetectWithRankLoss(model)
        logger.info("成功创建损失函数")
    except Exception as e:
        logger.error(f"创建损失函数失败: {e}")
        return False
    
    # 尝试进行前向传播
    try:
        with torch.no_grad():
            # 获取模型输出
            outputs = model(batch["img"])
            logger.info("模型前向传播成功")
            
            # 计算损失
            loss, loss_items = criterion(outputs, batch)
            logger.info(f"损失计算成功: 总损失={loss.item():.4f}")
            logger.info(f"损失组成: box={loss_items[0].item():.4f}, cls={loss_items[1].item():.4f}, dfl={loss_items[2].item():.4f}, rank={loss_items[3].item():.4f}")
            
            return True
    except Exception as e:
        logger.error(f"损失计算失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def analyze_ranking_data(batch):
    """详细分析批次中的排序相关数据"""
    logger.info("=" * 50)
    logger.info("分析排序相关数据")
    logger.info("=" * 50)
    
    # 检查必要字段是否存在
    required_fields = ["cluster_ids", "h_rel", "cls"]
    for field in required_fields:
        if field not in batch:
            logger.error(f"批次中缺少必要字段: {field}")
            return False
    
    # 获取数据
    cluster_ids = batch["cluster_ids"]
    h_rel = batch["h_rel"]
    classes = batch["cls"]
    
    # 检查数据形状
    logger.info(f"cluster_ids形状: {cluster_ids.shape}")
    logger.info(f"h_rel形状: {h_rel.shape}")
    logger.info(f"classes形状: {classes.shape}")
    
    # 展平数据以便分析
    cluster_ids_flat = cluster_ids.flatten()
    h_rel_flat = h_rel.flatten()
    classes_flat = classes.flatten()
    
    # 检查有效数据
    valid_mask = (cluster_ids_flat >= 0) & (h_rel_flat >= 0) & (h_rel_flat <= 1)
    valid_count = valid_mask.sum().item()
    logger.info(f"有效数据点数量: {valid_count}/{len(cluster_ids_flat)}")
    
    if valid_count == 0:
        logger.warning("没有有效的排序数据点")
        return False
    
    # 获取有效数据
    valid_cluster_ids = cluster_ids_flat[valid_mask]
    valid_h_rel = h_rel_flat[valid_mask]
    valid_classes = classes_flat[valid_mask]
    
    # 分析串ID分布
    unique_clusters = torch.unique(valid_cluster_ids)
    logger.info(f"不同的串ID数量: {len(unique_clusters)}")
    logger.info(f"串ID列表: {unique_clusters.tolist()}")
    
    # 分析每个串
    pair_count = 0
    valid_pair_count = 0
    
    for cluster_id in unique_clusters:
        # 跳过ID为0的串（通常是背景或无效串）
        if cluster_id == 0:
            continue
            
        # 找到当前串的所有番茄
        cluster_mask = valid_cluster_ids == cluster_id
        cluster_h_rel = valid_h_rel[cluster_mask]
        cluster_classes = valid_classes[cluster_mask]
        
        n_tomatoes = len(cluster_h_rel)
        logger.info(f"串ID {int(cluster_id.item())}: 包含 {n_tomatoes} 个番茄")
        
        if n_tomatoes <= 1:
            logger.info(f"  - 跳过: 只有一个番茄")
            continue  # 跳过只有一个番茄的串
        
        # 打印串内番茄信息
        logger.info(f"  - 高度: {cluster_h_rel.tolist()}")
        logger.info(f"  - 类别: {cluster_classes.tolist()}")
        
        # 比较同一串内的所有番茄对
        for i in range(n_tomatoes):
            for j in range(i + 1, n_tomatoes):
                h1, h2 = cluster_h_rel[i], cluster_h_rel[j]
                cls1, cls2 = cluster_classes[i], cluster_classes[j]
                
                # 计算高度差异
                h_diff = torch.abs(h1 - h2)
                pair_count += 1
                
                logger.info(f"  - 对比: 番茄{i}(高度={h1.item():.2f}, 类别={int(cls1.item())}) vs 番茄{j}(高度={h2.item():.2f}, 类别={int(cls2.item())})")
                logger.info(f"    高度差异: {h_diff.item():.4f}")
                
                # 检查高度差异是否显著
                margin = 0.1  # 默认间隔阈值
                if h_diff > margin:
                    # 确定预期的排序关系
                    higher_tomato, lower_tomato = (i, j) if h1 < h2 else (j, i)
                    higher_cls = cluster_classes[higher_tomato]
                    lower_cls = cluster_classes[lower_tomato]
                    
                    logger.info(f"    高度差异显著 (>{margin})")
                    logger.info(f"    较高番茄: 索引={higher_tomato}, 类别={int(higher_cls.item())}")
                    logger.info(f"    较低番茄: 索引={lower_tomato}, 类别={int(lower_cls.item())}")
                    
                    # 检查类别关系
                    if higher_cls < lower_cls:
                        logger.info(f"    ✓ 成熟度排序与高度排序一致")
                        valid_pair_count += 1
                    elif higher_cls > lower_cls:
                        logger.info(f"    ✗ 成熟度排序与高度排序不一致")
                    else:
                        logger.info(f"    - 相同类别")
                else:
                    logger.info(f"    高度差异不显著 (<={margin})")
    
    logger.info(f"总计番茄对数量: {pair_count}")
    logger.info(f"有效排序对数量: {valid_pair_count}")
    
    return True

def main():
    """主函数"""
    # 设置参数
    data_yaml = "tomato_data.yaml"
    model_yaml = os.path.join(YOLO_PATH, "ultralytics/cfg/models/v12/yolo12-A2C2f-DYT-EfficientHead-MambaOut.yaml")
    batch_size = 4
    img_size = 640
    
    # 检查文件是否存在
    if not os.path.exists(data_yaml):
        logger.error(f"数据配置文件不存在: {data_yaml}")
        sys.exit(1)
    
    if not os.path.exists(model_yaml):
        logger.error(f"模型配置文件不存在: {model_yaml}")
        sys.exit(1)
    
    # 加载数据配置
    try:
        with open(data_yaml, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        logger.info(f"成功加载数据配置: {data_yaml}")
    except Exception as e:
        logger.error(f"加载数据配置失败: {e}")
        sys.exit(1)
    
    # 确保数据配置包含必要的字段
    if "train" not in data_config:
        logger.error("数据配置缺少'train'字段")
        sys.exit(1)
    
    # 设置参数
    try:
        args = get_cfg()  # 使用默认配置
        args.data = data_yaml
        args.model = model_yaml
        args.batch = batch_size
        args.imgsz = img_size
        args.task = "tomato"  # 设置任务类型为番茄检测
        args.rect = False  # 不使用矩形训练
        args.cache = False  # 不使用缓存
        args.single_cls = False  # 不使用单类别
        args.augment = False  # 不使用数据增强
        args.stride = 32  # 步长
        args.pad = 0.0  # 填充
        args.loss = "TomatoDetectWithRankLoss(lambda_rank=0.2)"  # 设置损失函数
    except Exception as e:
        logger.error(f"创建配置失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    # 确保数据配置包含番茄特有标志
    data_config["has_cluster_ids"] = True
    data_config["has_h_rel"] = True
    data_config["has_cluster_id"] = True  # 兼容性
    
    # 创建数据集
    try:
        # 获取训练集路径
        train_path = data_config["train"]
        logger.info(f"训练集路径: {train_path}")
        
        # 检查路径是否存在
        if not os.path.exists(train_path):
            logger.error(f"训练集路径不存在: {train_path}")
            sys.exit(1)
        
        # 创建数据集
        dataset = TomatoYOLODataset(
            img_path=train_path,
            imgsz=args.imgsz,
            batch_size=args.batch,
            augment=args.augment,
            hyp=args,
            rect=args.rect,
            cache=args.cache,
            single_cls=args.single_cls,
            stride=args.stride,
            pad=args.pad,
            prefix='debug: ',
            data=data_config,  # 使用从YAML加载的配置
            args=args,  # 传递args参数
        )
        logger.info(f"创建数据集成功: {len(dataset)} 张图像")
        
        # 检查数据集属性
        logger.info(f"数据集属性: has_cluster_ids={dataset.has_cluster_ids}, has_h_rel={dataset.has_h_rel}")
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch, 
            shuffle=True, 
            collate_fn=TomatoYOLODataset.collate_fn,
            num_workers=0  # 不使用多进程加载
        )
        logger.info(f"创建数据加载器成功")
        
        # 获取一个批次
        batch = next(iter(dataloader))
        logger.info(f"获取批次成功: 批次大小 {batch['img'].shape[0]}")
        
        # 检查批次中的关键字段
        logger.info(f"批次字段: {list(batch.keys())}")
        
        # 打印边界框的格式和范围
        if 'bboxes' in batch:
            bboxes = batch['bboxes']
            logger.info(f"bboxes: 形状={bboxes.shape}, 类型={bboxes.dtype}")
            
            # 检查格式：计算宽度和高度
            w_range = (bboxes[:, 2] - bboxes[:, 0]).cpu().numpy()
            h_range = (bboxes[:, 3] - bboxes[:, 1]).cpu().numpy()
            
            # 如果大多数宽度和高度小于1，可能是归一化的边界框坐标(x1,y1,x2,y2)
            w_small = (w_range < 1).sum() / len(w_range) > 0.9
            h_small = (h_range < 1).sum() / len(h_range) > 0.9
            
            if w_small and h_small:
                logger.info(f"边界框可能是归一化的(x1,y1,x2,y2)格式")
                logger.info(f"边界框宽度范围: [{w_range.min():.4f}, {w_range.max():.4f}]")
                logger.info(f"边界框高度范围: [{h_range.min():.4f}, {h_range.max():.4f}]")
            else:
                # 如果宽度和高度值很大，那么可能是像素坐标，或者中心点+宽高表示
                # 在这种情况下，检查首个值是否都小于1（归一化中心点）
                first_coord_small = (bboxes[:, 0] < 1).float().mean() > 0.9 and (bboxes[:, 1] < 1).float().mean() > 0.9
                if first_coord_small:
                    logger.info(f"边界框可能是归一化的中心点(x,y,w,h)格式")
                else:
                    logger.info(f"边界框可能是像素坐标格式")
                
                logger.info(f"第一列范围: [{bboxes[:, 0].min():.4f}, {bboxes[:, 0].max():.4f}]")
                logger.info(f"第二列范围: [{bboxes[:, 1].min():.4f}, {bboxes[:, 1].max():.4f}]")
                logger.info(f"第三列范围: [{bboxes[:, 2].min():.4f}, {bboxes[:, 2].max():.4f}]")
                logger.info(f"第四列范围: [{bboxes[:, 3].min():.4f}, {bboxes[:, 3].max():.4f}]")
            
            # 打印一些边界框样本
            logger.info(f"边界框样本: {bboxes[:5].tolist()}")
        else:
            logger.warning("批次中没有bboxes字段")
        
        # 检查cluster_ids和h_rel字段
        if 'cluster_ids' in batch:
            logger.info(f"cluster_ids: 形状={batch['cluster_ids'].shape}, 类型={batch['cluster_ids'].dtype}")
            logger.info(f"cluster_ids样本: {batch['cluster_ids'][:5].tolist()}")
        else:
            logger.warning("批次中没有cluster_ids字段")
        
        if 'h_rel' in batch:
            logger.info(f"h_rel: 形状={batch['h_rel'].shape}, 类型={batch['h_rel'].dtype}")
            logger.info(f"h_rel样本: {batch['h_rel'][:5].tolist()}")
        else:
            logger.warning("批次中没有h_rel字段")
        
        # 可视化批次
        visualize_batch(batch)
        
        # 分析排序数据
        analyze_ranking_data(batch)
        
        # 加载模型
        try:
            logger.info(f"加载模型: {model_yaml}")
            model = TomatoDetectionModel(cfg=model_yaml, nc=data_config["nc"])
            logger.info("模型加载成功")
            
            # 检查模型输入
            check_model_input(model, batch)
            
            # 检查损失计算
            check_loss_calculation(model, batch)
            
        except Exception as e:
            logger.error(f"加载或检查模型失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
    except Exception as e:
        logger.error(f"创建数据集或获取批次失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
    logger.info("调试完成") 