#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
可视化模型对番茄排序和高度预测的结果
"""

import os
import sys
import torch
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

# 设置日志
log_dir = Path('debug_logs')
log_dir.mkdir(exist_ok=True)
log_file = log_dir / 'rank_vis.log'

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('rank_vis')

# 添加YOLO路径
YOLO_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, YOLO_PATH)

try:
    # 应用修复补丁
    from tomato_model_fix_patch import apply_patches
    apply_patches()
    
    # 导入必要的模块
    from ultralytics.nn.tasks import TomatoDetectionModel
    from ultralytics.utils.loss import TomatoDetectWithRankLoss
    logger.info("成功应用补丁并导入YOLO模块")
except ImportError as e:
    logger.error(f"导入模块失败: {e}")
    sys.exit(1)

def load_model():
    """加载番茄检测模型"""
    model_yaml = os.path.join(YOLO_PATH, "ultralytics/cfg/models/v12/yolo12-A2C2f-DYT-EfficientHead-MambaOut.yaml")
    
    # 加载模型
    logger.info(f"加载模型: {model_yaml}")
    model = TomatoDetectionModel(cfg=model_yaml, nc=6)
    logger.info("模型加载成功")
    
    # 为模型添加缺少的args属性
    class Args:
        def __init__(self):
            self.box = 7.5  # 边界框损失权重
            self.cls = 0.5  # 分类损失权重
            self.dfl = 1.5  # DFL损失权重
            self.device = next(model.parameters()).device
    
    # 添加args属性
    model.args = Args()
    
    # 修复模型数据类型问题
    for name, buffer in model.named_buffers():
        if buffer.dtype != torch.float32:
            logger.info(f"转换buffer {name} 从 {buffer.dtype} 到 float32")
            buffer.data = buffer.data.to(torch.float32)
    
    # 切换到评估模式
    model.eval()
    
    return model

def create_test_image(batch_size=1, num_clusters=2, cluster_size=3, image_size=640):
    """
    创建一个测试图像，包含番茄串
    
    Args:
        batch_size: 批次大小
        num_clusters: 每张图像中番茄串的数量
        cluster_size: 每个串中番茄的数量
        image_size: 图像尺寸
    
    Returns:
        batch: 批次数据
        images: 原始图像（带可视化的番茄串）
    """
    # 创建空白图像
    images = []
    
    # 生成颜色映射（从绿色到红色）
    cmap = LinearSegmentedColormap.from_list(
        "tomato_ripeness", 
        [(0.0, (0.0, 0.8, 0.0)),    # 绿色 (class 3)
         (0.3, (0.5, 0.8, 0.0)),    # 黄绿色 (class 2)
         (0.7, (1.0, 0.5, 0.0)),    # 橙色 (class 1)
         (1.0, (1.0, 0.0, 0.0))],   # 红色 (class 0)
    )
    
    # 准备批次数据
    batch = {
        'img': torch.zeros(batch_size, 3, image_size, image_size),
        'batch_idx': [],
        'cls': [],
        'bboxes': [],
        'cluster_ids': [],
        'h_rel': []
    }
    
    total_objects = 0
    
    for b in range(batch_size):
        # 创建空白RGB图像
        img = np.ones((image_size, image_size, 3), dtype=np.uint8) * 220
        
        # 为每个串分配一个位置
        cluster_positions = []
        for _ in range(num_clusters):
            # 随机选择串的x坐标
            x = np.random.randint(image_size // 4, 3 * image_size // 4)
            cluster_positions.append(x)
        
        # 为每个串生成番茄
        for c, x in enumerate(cluster_positions):
            cluster_id = c + 1  # 串ID从1开始
            
            # 决定这个串的排列方式：0=成熟度从上到下递增，1=随机，2=从上到下递减
            arrangement = np.random.choice([0, 1, 2], p=[0.6, 0.2, 0.2])
            
            # 创建番茄的类别（成熟度）
            if arrangement == 0:
                # 从上到下递减成熟度（上面较成熟，下面较不成熟）
                classes = np.arange(cluster_size)
            elif arrangement == 1:
                # 随机成熟度
                classes = np.random.randint(0, 4, size=cluster_size)
            else:
                # 从上到下递增成熟度（上面较不成熟，下面较成熟）
                classes = np.arange(cluster_size)[::-1]
            
            # 确保类别在0-3范围内
            classes = np.clip(classes, 0, 3)
            
            # 计算番茄的位置
            y_spacing = image_size // (cluster_size + 1)
            y_positions = [y_spacing * (i + 1) for i in range(cluster_size)]
            
            # 在图像上绘制番茄
            for i, y in enumerate(y_positions):
                cls = classes[i]
                
                # 番茄大小（更成熟的番茄略大）
                radius = int(20 + (3-cls) * 5)
                
                # 计算相对高度（0=顶部，1=底部）
                h_rel = y / image_size
                
                # 获取番茄颜色
                color_norm = (3 - cls) / 3  # 转换为0-1范围，越成熟值越大
                color = cmap(color_norm)
                color_rgb = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
                
                # 绘制番茄
                cv2.circle(img, (x, y), radius, color_rgb, -1)
                
                # 添加类别标签
                cls_name = ['fully ripe', 'ripe', 'turning', 'green'][cls]
                cv2.putText(img, cls_name, (x + radius + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # 添加h_rel标签
                cv2.putText(img, f"h={h_rel:.2f}", (x - radius - 60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # 添加数据到批次
                batch['batch_idx'].append(b)
                batch['cls'].append([cls])
                
                # 边界框（xc, yc, w, h）格式
                box_size = radius * 2.2  # 略大于番茄实际大小
                batch['bboxes'].append([x / image_size, y / image_size, 
                                       box_size / image_size, box_size / image_size])
                batch['cluster_ids'].append([cluster_id])
                batch['h_rel'].append([h_rel])
                
                total_objects += 1
        
        # 添加串编号
        for c, x in enumerate(cluster_positions):
            cv2.putText(img, f"Cluster {c+1}", (x - 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # 转换图像为PyTorch格式并添加到批次
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        batch['img'][b] = img_tensor
        
        # 保存原始图像
        images.append(img)
    
    # 转换列表为张量
    batch['batch_idx'] = torch.tensor(batch['batch_idx'])
    batch['cls'] = torch.tensor(batch['cls'])
    batch['bboxes'] = torch.tensor(batch['bboxes'])
    batch['cluster_ids'] = torch.tensor(batch['cluster_ids'])
    batch['h_rel'] = torch.tensor(batch['h_rel'])
    
    logger.info(f"创建了测试批次: 批次大小={batch_size}, 总对象数量={total_objects}")
    
    return batch, images

def run_model(model, batch):
    """运行模型并获取预测结果"""
    with torch.no_grad():
        outputs = model(batch['img'])
    
    logger.info("模型前向传播完成")
    
    return outputs

def extract_h_pos_features(outputs):
    """从模型输出中提取高度特征"""
    if isinstance(outputs, tuple) and len(outputs) > 1 and isinstance(outputs[1], dict):
        h_pos_features = outputs[1].get('h_pos', None)
        if h_pos_features is not None:
            logger.info(f"成功提取高度特征: 特征数量={len(h_pos_features)}")
            logger.info(f"高度特征形状: {[h.shape for h in h_pos_features]}")
            return h_pos_features
    
    logger.warning("未找到高度特征")
    return None

def visualize_h_pos_features(h_pos_features, images, save_dir='debug_logs'):
    """可视化高度特征预测"""
    if h_pos_features is None or len(h_pos_features) == 0:
        logger.error("没有高度特征可视化")
        return
    
    # 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # 设置图表大小
    plt.figure(figsize=(16, 6 * len(images)))
    
    # 为每张图像可视化高度特征
    for batch_idx, img in enumerate(images):
        # 原始图像
        plt.subplot(len(images), 3, batch_idx * 3 + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"原始图像 {batch_idx+1}")
        plt.axis('off')
        
        # 选择一个合适的特征层可视化（中间分辨率）
        h_pos = h_pos_features[1][batch_idx, 0].cpu().numpy()
        
        # 可视化高度预测
        plt.subplot(len(images), 3, batch_idx * 3 + 2)
        plt.imshow(h_pos, cmap='plasma', vmin=0, vmax=1)
        plt.colorbar(label='预测高度')
        plt.title(f"高度预测图 (原始)")
        plt.axis('off')
        
        # 将高度特征上采样到原图尺寸
        h, w = img.shape[:2]
        h_pos_resized = cv2.resize(h_pos, (w, h))
        
        # 叠加在原图上
        plt.subplot(len(images), 3, batch_idx * 3 + 3)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.imshow(h_pos_resized, cmap='plasma', alpha=0.6, vmin=0, vmax=1)
        plt.colorbar(label='预测高度')
        plt.title(f"高度预测叠加图")
        plt.axis('off')
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(save_dir / "height_prediction.png")
    logger.info(f"高度预测可视化已保存至: {save_dir}/height_prediction.png")
    
    # 关闭图表
    plt.close()

def analyze_height_predictions(h_pos_features, batch):
    """分析高度预测是否符合期望（上小下大）"""
    if h_pos_features is None or len(h_pos_features) == 0:
        logger.error("没有高度特征可分析")
        return
    
    # 分析垂直方向上的高度渐变
    for level, h_pos in enumerate(h_pos_features):
        batch_size, _, h, w = h_pos.shape
        
        # 计算垂直渐变
        for b in range(batch_size):
            feature = h_pos[b, 0].cpu().numpy()
            
            # 计算垂直差异（下 - 上）
            vert_diff = feature[1:, :] - feature[:-1, :]
            
            # 计算违反单调递增的比例
            violation_ratio = (vert_diff < 0).mean()
            
            logger.info(f"批次 {b+1}, 特征层级 {level+1}: 违反递增规则的像素比例: {violation_ratio:.2%}")
            
            # 总体评估
            if violation_ratio < 0.3:
                logger.info(f"批次 {b+1}, 特征层级 {level+1}: 高度预测总体符合从上到下递增趋势 ✅")
            else:
                logger.info(f"批次 {b+1}, 特征层级 {level+1}: 高度预测不太符合从上到下递增趋势 ❌")

def main():
    """主函数"""
    logger.info("开始番茄排序可视化测试...")
    
    # 加载模型
    model = load_model()
    
    # 创建测试图像
    batch, images = create_test_image(batch_size=2, num_clusters=2, cluster_size=4)
    
    # 运行模型
    outputs = run_model(model, batch)
    
    # 提取高度特征
    h_pos_features = extract_h_pos_features(outputs)
    
    # 可视化高度特征
    visualize_h_pos_features(h_pos_features, images)
    
    # 分析高度预测
    analyze_height_predictions(h_pos_features, batch)
    
    logger.info("番茄排序可视化测试完成")

if __name__ == "__main__":
    main() 