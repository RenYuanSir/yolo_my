#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import cv2
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')

# 创建日志目录
log_dir = Path('debug_logs')
log_dir.mkdir(exist_ok=True)
log_file = log_dir / 'debug_loss.log'

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('debug_loss')

# 添加YOLO路径
YOLO_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, YOLO_PATH)

# 导入YOLO相关模块
try:
    from ultralytics.utils.loss import TomatoDetectWithRankLoss
    from ultralytics.nn.tasks import TomatoDetectionModel
    from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors
    logger.info("成功导入YOLO模块")
except ImportError as e:
    logger.error(f"导入YOLO模块失败: {e}")
    sys.exit(1)

def load_model():
    """加载模型并准备相关组件"""
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
    for buffer in model.buffers():
        if buffer.dtype != torch.float32:
            buffer.data = buffer.data.to(torch.float32)
    
    # 创建损失计算器
    loss_fn = TomatoDetectWithRankLoss(model)
    logger.info(f"损失函数: loss_fn.no = {loss_fn.no}, loss_fn.reg_max = {loss_fn.reg_max}, loss_fn.nc = {loss_fn.nc}")
    
    return model, loss_fn

def debug_fg_mask(loss_fn, preds, batch):
    """详细调试fg_mask的生成过程"""
    logger.info("=" * 80)
    logger.info("开始调试fg_mask生成过程...")
    
    # 1. 规范化预测并提取特征
    feats, pred_h_pos = loss_fn._normalize_predictions(preds)
    if feats is None:
        logger.error("特征为空，无法继续调试")
        return False
        
    # 2. 从特征中提取预测分布和分数
    pred_distri, pred_scores = loss_fn._extract_predictions(feats)
    logger.info(f"预测分布形状: {pred_distri.shape}, 预测分数形状: {pred_scores.shape}")
    
    # 3. 获取批次大小和图像大小
    batch_size = pred_scores.shape[0]
    imgsz = torch.tensor(feats[0].shape[2:], device=loss_fn.device, dtype=pred_scores.dtype) * loss_fn.stride[0]
    logger.info(f"批次大小: {batch_size}, 图像尺寸: {imgsz}")
    
    # 4. 创建锚点网格和步长张量
    anchor_points, stride_tensor = make_anchors(feats, loss_fn.stride, 0.5)
    logger.info(f"锚点数量: {anchor_points.shape[0]}, 步长张量形状: {stride_tensor.shape}")
    logger.info(f"锚点示例: {anchor_points[:5]}, 步长示例: {stride_tensor[:5]}")
    
    # 检查锚点维度
    logger.info(f"锚点维度: {anchor_points.dim()}, 形状: {anchor_points.shape}")
    logger.info(f"步长张量维度: {stride_tensor.dim()}, 形状: {stride_tensor.shape}")
    
    # 5. 准备目标数据
    targets = loss_fn._prepare_targets(batch, batch_size, imgsz)
    logger.info(f"目标形状: {targets.shape}")
    logger.info(f"目标示例: {targets[0, 0]}")
    
    # 6. 分离标签和边界框
    gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)  # 非零边界框的掩码
    logger.info(f"标签形状: {gt_labels.shape}, 边界框形状: {gt_bboxes.shape}, 掩码形状: {mask_gt.shape}")
    logger.info(f"目标真值掩码统计: 总数={mask_gt.numel()}, 真值数量={mask_gt.sum().item()}, 占比={mask_gt.sum().item()/mask_gt.numel()*100:.2f}%")
    
    # 7. 将预测边界框形式转换为XYXY格式
    pred_bboxes = loss_fn.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
    logger.info(f"预测边界框形状: {pred_bboxes.shape}")
    
    # 8. 分配正样本
    try:
        logger.info("开始正样本分配...")
        with torch.cuda.amp.autocast(enabled=False):
            target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = loss_fn.assigner(
                pred_scores.detach().sigmoid(),
                pred_bboxes.detach() * stride_tensor,
                anchor_points * stride_tensor,
                gt_labels,
                gt_bboxes,
                mask_gt
            )
        
        # 输出fg_mask信息
        logger.info(f"fg_mask形状: {fg_mask.shape}, 类型: {fg_mask.dtype}, 设备: {fg_mask.device}")
        logger.info(f"fg_mask统计: 总数={fg_mask.numel()}, 正样本数量={fg_mask.sum().item()}, 正样本占比={fg_mask.sum().item()/fg_mask.numel()*100:.2f}%")
        logger.info(f"fg_mask按批次统计:")
        for i in range(batch_size):
            num_pos = fg_mask[i].sum().item()
            logger.info(f"  批次 {i}: 正样本数量={num_pos}, 占比={num_pos/fg_mask[i].numel()*100:.2f}%")
            
        # 检查assigner内部各个步骤的输出
        logger.info("检查TaskAlignedAssigner内部计算...")
        num_classes = loss_fn.nc
        topk = loss_fn.assigner.topk  # 通常为10
        beta = loss_fn.assigner.beta  # 通常为6.0
        
        bs, n = pred_scores.shape[:2]
        if fg_mask.sum() == 0:
            logger.warning("❌ 没有正样本，先进行一些粗略检查...")
            # 检查预测分数的值域
            logger.info(f"预测分数范围: [{pred_scores.min().item():.4f}, {pred_scores.max().item():.4f}]")
            logger.info(f"预测分数sigmoid后范围: [{torch.sigmoid(pred_scores).min().item():.4f}, {torch.sigmoid(pred_scores).max().item():.4f}]")
            
            # 检查预测框与GT框的IoU
            ious = torch.zeros((batch_size, n, mask_gt.sum()), device=pred_scores.device)
            if mask_gt.sum() > 0:  # 确保有GT框
                # 找到mask_gt为True的索引
                b_idx, t_idx = torch.where(mask_gt.squeeze(-1))
                # 计算预测框与GT框的IoU
                for i in range(batch_size):
                    batch_gt_idx = (b_idx == i)
                    if batch_gt_idx.sum() > 0:
                        batch_t_idx = t_idx[batch_gt_idx]
                        for j, idx in enumerate(batch_t_idx):
                            gt_box = gt_bboxes[i, idx]
                            pred_box = pred_bboxes[i] * stride_tensor
                            # 计算IoU
                            iou = box_iou(gt_box.unsqueeze(0), pred_box)[0]
                            logger.info(f"批次 {i}, GT框 {idx}: 最大IoU={iou.max().item():.4f}, 平均IoU={iou.mean().item():.4f}")
                            
            logger.warning("正样本分配失败，需要手动构造一些正样本来测试损失计算...")
            
            # 模拟一些正样本
            num_pos_per_batch = 10  # 每个批次模拟的正样本数量
            for i in range(batch_size):
                # 随机选择一些点作为正样本
                pos_indices = torch.randperm(n)[:num_pos_per_batch]
                fg_mask[i, pos_indices] = True
            
            logger.info(f"模拟后的fg_mask统计: 总数={fg_mask.numel()}, 正样本数量={fg_mask.sum().item()}, 正样本占比={fg_mask.sum().item()/fg_mask.numel()*100:.2f}%")
                
        return fg_mask, anchor_points, stride_tensor, gt_bboxes, target_bboxes, pred_bboxes
    
    except Exception as e:
        logger.error(f"分配正样本时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None, None, None, None

def visualize_anchors_targets(fg_mask, anchor_points, stride_tensor, gt_bboxes, target_bboxes, pred_bboxes, save_dir='debug_viz'):
    """可视化锚点和目标分布"""
    # 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    batch_size = fg_mask.shape[0]
    
    for i in range(batch_size):
        # 创建大小为640x640的图像
        img_size = 640
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
        
        # 绘制GT边界框
        gt_valid = gt_bboxes[i].sum(dim=1) > 0
        gt_boxes = gt_bboxes[i, gt_valid].cpu().numpy()
        for box in gt_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色表示GT框
        
        # 绘制正样本锚点
        pos_mask = fg_mask[i]
        pos_anchors = anchor_points[pos_mask].cpu().numpy()
        pos_strides = stride_tensor[pos_mask].cpu().numpy()
        
        # 缩放点以匹配图像尺寸
        pos_anchors_scaled = pos_anchors * img_size
        
        for point, stride in zip(pos_anchors_scaled, pos_strides):
            x, y = map(int, point)
            radius = int(stride / 2)  # 根据stride调整点的大小
            cv2.circle(img, (x, y), radius, (0, 0, 255), -1)  # 红色表示正样本锚点
            
        # 绘制预测框
        if pred_bboxes is not None:
            pos_pred_boxes = (pred_bboxes[i][pos_mask] * stride_tensor[pos_mask].view(-1, 1)).cpu().numpy()
            for box in pos_pred_boxes:
                x1, y1, x2, y2 = map(int, box * img_size / 640)  # 缩放到图像大小
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)  # 蓝色表示预测框
        
        # 保存图像
        cv2.imwrite(str(save_dir / f'batch_{i}_anchors_targets.png'), img)
        logger.info(f"已保存可视化结果: {save_dir / f'batch_{i}_anchors_targets.png'}")

def debug_rank_loss(loss_fn, preds, batch):
    """调试rank_loss的计算过程"""
    logger.info("=" * 80)
    logger.info("开始调试rank_loss...")
    
    # 调用损失函数的compute_ranking_loss方法
    try:
        rank_loss = loss_fn.compute_ranking_loss(batch)
        logger.info(f"rank_loss计算结果: {rank_loss.item() if rank_loss is not None else None}")
        
        # 检查输入
        logger.info(f"batch中的键: {list(batch.keys())}")
        if 'cluster_ids' in batch:
            logger.info(f"cluster_ids形状: {batch['cluster_ids'].shape}, 类型: {batch['cluster_ids'].dtype}")
            logger.info(f"cluster_ids内容: {batch['cluster_ids']}")
        else:
            logger.warning("batch中没有cluster_ids")
            
        if 'h_rel' in batch:
            logger.info(f"h_rel形状: {batch['h_rel'].shape}, 类型: {batch['h_rel'].dtype}")
            logger.info(f"h_rel内容: {batch['h_rel']}")
        else:
            logger.warning("batch中没有h_rel")
            
        # 检查compute_ranking_loss中是否真正使用了输入
        # 通过在batch中添加一些极端值来测试
        for i in range(len(batch['h_rel'])):
            if i < len(batch['h_rel']) // 2:
                batch['h_rel'][i] = 0.0  # 设为最小值
            else:
                batch['h_rel'][i] = 1.0  # 设为最大值
        
        # 再次调用计算
        new_rank_loss = loss_fn.compute_ranking_loss(batch)
        logger.info(f"修改h_rel后的rank_loss: {new_rank_loss.item() if new_rank_loss is not None else None}")
        
        # 检查是否对修改有响应
        if rank_loss is not None and new_rank_loss is not None:
            if abs(rank_loss.item() - new_rank_loss.item()) < 1e-6:
                logger.warning("❌ rank_loss对h_rel的修改没有响应，可能返回的是常数")
            else:
                logger.info("✅ rank_loss对h_rel的修改有响应")
        
        return rank_loss
        
    except Exception as e:
        logger.error(f"计算rank_loss时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def create_sanity_check_sample():
    """创建一个简单的正样本测试用例"""
    logger.info("=" * 80)
    logger.info("创建简单的正样本测试用例...")
    
    # 创建一个简单的批次
    batch_size = 1
    img = torch.rand(batch_size, 3, 640, 640)
    
    # 创建边界框: cx, cy, w, h 格式，值为归一化的(0-1)
    bboxes = torch.tensor([
        [0.5, 0.5, 0.2, 0.2],  # 中心点
        [0.2, 0.2, 0.1, 0.1],  # 左上角
        [0.8, 0.8, 0.1, 0.1],  # 右下角
    ], dtype=torch.float32)
    
    # 创建标签，使用简单的类别
    cls = torch.zeros((3, 1), dtype=torch.float32)  # 所有样本都是类别0
    
    # 创建批次索引
    batch_idx = torch.zeros(3, dtype=torch.float32)  # 所有框都属于第0个样本
    
    # 创建cluster_ids: 给每个框分配一个组
    cluster_ids = torch.tensor([[0], [1], [1]], dtype=torch.float32)
    
    # 创建h_rel: 高度关系，用于排序损失
    h_rel = torch.tensor([[0.5], [0.1], [0.9]], dtype=torch.float32)
    
    # 组合为batch
    batch = {
        'img': img,
        'batch_idx': batch_idx,
        'cls': cls,
        'bboxes': bboxes,
        'cluster_ids': cluster_ids,
        'h_rel': h_rel,
    }
    
    return batch

def box_iou(box1, box2):
    """计算两组框的IoU"""
    # 计算交集
    inter = _box_inter(box1, box2)
    
    # 计算并集
    area1 = _box_area(box1)
    area2 = _box_area(box2)
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter
    
    # 计算IoU
    iou = inter / (union + 1e-6)
    return iou

def _box_area(box):
    """计算框的面积"""
    return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])

def _box_inter(box1, box2):
    """计算两组框的交集"""
    # 计算交集的左上角和右下角
    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])
    
    # 计算交集的宽高
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    return inter

def main():
    """主函数"""
    logger.info("开始调试损失计算...")
    
    # 加载模型和损失函数
    model, loss_fn = load_model()
    
    # 准备测试输入
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 640, 640)
    
    # 设置为评估模式
    model.eval()
    
    # 执行前向传播
    with torch.no_grad():
        logger.info("执行前向传播...")
        preds = model(dummy_input)
        logger.info("前向传播成功")
    
    # 创建批次数据
    batch = {
        'img': dummy_input,
        'batch_idx': torch.zeros(10),
        'cls': torch.zeros(10, 1),
        'bboxes': torch.rand(10, 4),  # XYWH格式，值在0-1之间
        'cluster_ids': torch.zeros(10, 1),
        'h_rel': torch.rand(10, 1),
    }
    
    # 步骤1：检查fg_mask
    fg_mask, anchor_points, stride_tensor, gt_bboxes, target_bboxes, pred_bboxes = debug_fg_mask(loss_fn, preds, batch)
    
    # 步骤2：可视化anchors和targets
    if fg_mask is not None:
        visualize_anchors_targets(fg_mask, anchor_points, stride_tensor, gt_bboxes, target_bboxes, pred_bboxes)
    
    # 步骤3：检查rank_loss
    rank_loss = debug_rank_loss(loss_fn, preds, batch)
    
    # 步骤4：创建简单的正样本测试用例
    sanity_batch = create_sanity_check_sample()
    
    # 使用简单测试用例计算损失
    logger.info("=" * 80)
    logger.info("使用简单测试用例计算损失...")
    
    try:
        # 计算损失
        loss, loss_items = loss_fn(preds, sanity_batch)
        
        # 打印损失结果
        logger.info(f"总损失: {loss.item()}")
        logger.info(f"损失项: box_loss={loss_items[0].item()}, cls_loss={loss_items[1].item()}, dfl_loss={loss_items[2].item()}, rank_loss={loss_items[3].item()}")
        logger.info("简单测试用例损失计算成功")
    except Exception as e:
        logger.error(f"简单测试用例计算损失失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("调试完成")

if __name__ == "__main__":
    main() 