# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import numpy as np
import os
import json
from pathlib import Path
import torch.nn.functional as F
import time
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import LOGGER, ops, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, TomatoMetrics, box_iou , bbox_iou
from ultralytics.utils.torch_utils import smart_inference_mode, de_parallel, select_device
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.ops import Profile
from ultralytics.utils.checks import check_imgsz
from ultralytics.data.utils import check_det_dataset


class TomatoValidator(DetectionValidator):
    """
    TomatoValidator类用于验证番茄检测模型，继承自DetectionValidator。
    
    添加了对h_pos预测的评估和排序指标的计算。
    
    Example:
        ```python
        from ultralytics.models.yolo.tomato import TomatoValidator

        args = dict(model="yolov12-tomato.pt", data="tomato.yaml")
        validator = TomatoValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """初始化番茄检测验证器，添加特定于番茄的指标。"""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = "tomato"  # 设置任务为tomato
        # 确保self.stats是一个列表字典，而不是索引字典
        self.stats = {'tp': [], 'conf': [], 'pred_cls': [], 'target_cls': []}
        

    def init_metrics(self, model):
        """初始化评估指标。"""
        super().init_metrics(model)
        # 使用TomatoMetrics替代DetMetrics
        self.model = model
        self.names = model.names
        self.nc = len(model.names)
        self.metrics = TomatoMetrics(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot, names=self.names)
        self.metrics.names = self.names
        self.metrics.nc = self.nc
        # 确保self.stats是一个列表字典，而不是索引字典
        self.stats = {'tp': [], 'conf': [], 'pred_cls': [], 'target_cls': []}
        
    def preprocess(self, batch):
        """预处理输入批次数据。"""
        # 首先确保batch["img"]被正确处理
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        
        # 将标签数据移至设备
        for k in ["batch_idx", "cls", "bboxes"]:
            if k in batch:
                batch[k] = batch[k].to(self.device)
        
        # 确保h_rel字段被转移到正确的设备
        if 'h_rel' in batch:
            batch['h_rel'] = batch['h_rel'].to(self.device)
            
        # 确保cluster_ids字段被转移到正确的设备
        if 'cluster_ids' in batch:
            batch['cluster_ids'] = batch['cluster_ids'].to(self.device)
        
        # 检查边界框是否为归一化坐标，如果是则转换为像素级坐标
        if 'bboxes' in batch and batch['bboxes'].numel() > 0:
            # 检查是否是归一化坐标（最大值<=1.0）
            if batch['bboxes'].max() <= 1.0:
                # 获取图像尺寸
                img_size = batch['img'].shape[2:]  # (h, w)
                # 将归一化坐标转换为像素级坐标
                # 注意：边界框格式为xywh，所以x,w对应宽度，y,h对应高度
                scale_factor = torch.tensor([img_size[1], img_size[0], img_size[1], img_size[0]], 
                                           device=batch['bboxes'].device)
                batch['bboxes'] = batch['bboxes'] * scale_factor
                
                # =================== CHECKPOINT VALIDATE ===================
                print("\n" + "="*50)
                print("🎯 CHECKPOINT VALIDATE: 边界框坐标转换")
                print(f"    图像尺寸: {img_size}")
                print(f"    转换前边界框范围: [0, 1]")
                print(f"    转换后边界框范围: [{batch['bboxes'].min().item():.4f}, {batch['bboxes'].max().item():.4f}]")
                print(f"    边界框形状: {batch['bboxes'].shape}")
                print("="*50 + "\n")
                # ====================================================
            
        # 检查批次是否为空
        if 'bboxes' in batch and batch['bboxes'].shape[0] == 0:
            LOGGER.warning(f"检测到空批次，添加假目标以避免错误")
            self._add_dummy_targets(batch)
            
        return batch
    
    def _add_dummy_targets(self, batch):
        """为空批次添加假目标以避免错误。"""
        bs = batch['img'].shape[0]  # 批次大小
        device = batch['img'].device
        
        # 创建假边界框和类别
        fake_bboxes = torch.zeros((1, 4), device=device)  # 一个空边界框
        fake_cls = torch.zeros((1, 1), device=device)  # 一个类别为0的标签
        fake_batch_idx = torch.zeros((1,), device=device)  # 批次索引为0
        
        # 如果是番茄数据集，添加番茄特有字段
        if 'cluster_ids' in batch:
            fake_cluster_ids = torch.zeros((1, 1), device=device)
            batch['cluster_ids'] = fake_cluster_ids
        if 'h_rel' in batch:
            fake_h_rel = torch.zeros((1, 1), device=device)
            batch['h_rel'] = fake_h_rel
            
        # 更新批次
        batch['bboxes'] = fake_bboxes
        batch['cls'] = fake_cls
        batch['batch_idx'] = fake_batch_idx
        
    def safe_compute_loss(self, model, batch, preds):
        """
        安全地计算损失，处理可能的错误和异常情况。
        
        Args:
            model: 模型对象
            batch: 批次数据
            preds: 预测结果
            
        Returns:
            loss: 损失值
            loss_items: 损失项
        """
        try:
            # 获取损失函数
            loss_fn = getattr(model, 'loss', None)
            
            # =================== CHECKPOINT VALIDATE ===================
            print("\n" + "="*50)
            print("🎯 CHECKPOINT VALIDATE: Loss Calculation")
            if 'bboxes' in batch and batch['bboxes'].numel() > 0:
                print(f"    batch['bboxes'] shape: {batch['bboxes'].shape}")
                # 判断是归一化坐标还是像素坐标
                if batch['bboxes'].max() > 1.0:
                    print(f"    batch['bboxes'] 格式: xywh (像素级)")
                else:
                    print(f"    batch['bboxes'] 格式: xywh (归一化)")
                print(f"    batch['bboxes'] 范围: [{batch['bboxes'].min().item():.4f}, {batch['bboxes'].max().item():.4f}]")
            
            if isinstance(preds, dict):
                print(f"    preds keys: {list(preds.keys())}")
            
            print(f"    当前模型任务类型: {getattr(model, 'task', 'unknown')}")
            print(f"    模型类型: {type(model).__name__}")
            print(f"    损失函数类型: {type(loss_fn).__name__ if loss_fn else 'None'}")
            print("="*50 + "\n")
            # ====================================================
            
            # 设置模型任务为tomato
            original_task = getattr(model, 'task', None)
            model.task = 'tomato'
            
            # 确保preds格式正确
            if not isinstance(preds, dict):
                # 如果preds不是字典格式，尝试转换为标准格式
                if isinstance(preds, tuple) and len(preds) == 2 and isinstance(preds[1], dict):
                    # 处理(output, dict)格式
                    if 'features' in preds[1]:
                        preds = {
                            'features': preds[1]['features'],
                            'h_pos': preds[1].get('h_pos', preds[1].get('h_rel', None))
                        }
                    else:
                        # 无法处理的格式
                        LOGGER.warning("无法识别的preds格式，无法计算损失")
                        return torch.zeros(1, device=self.device), torch.zeros(3, device=self.device)
            
            # 确保键名正确
            if 'h_rel' in preds and 'h_pos' not in preds:
                preds['h_pos'] = preds.pop('h_rel')
                print("    将preds中的'h_rel'键改为'h_pos'")
            
            # 计算损失
            try:
                # 直接传入模型的损失函数
                loss = model.loss(batch, preds)
                
                # =================== CHECKPOINT VALIDATE ===================
                print("\n" + "="*50)
                print("🎯 CHECKPOINT VALIDATE: Loss Result")
                print(f"    loss计算成功: {loss}")
                print("="*50 + "\n")
                # ====================================================
                
                # 解包损失
                if isinstance(loss, tuple) and len(loss) >= 2:
                    loss_items = loss[1]
                    loss = loss[0]
                else:
                    loss_items = torch.zeros(3, device=self.device)  # box, cls, dfl
                print(f"    loss_items: {loss_items.tolist()}")
                # 返回损失
                return loss, loss_items
            except Exception as e:
                LOGGER.warning(f"计算损失时发生错误: {e}")
                import traceback
                LOGGER.warning(f"错误详情: {traceback.format_exc()}")
                
                # 恢复模型的原始任务
                if original_task is not None:
                    model.task = original_task
                
                # 返回零损失
                return torch.zeros(1, device=self.device), torch.zeros(3, device=self.device)
        except Exception as e:
            LOGGER.warning(f"计算损失时发生错误: {e}")
            import traceback
            LOGGER.warning(f"错误详情: {traceback.format_exc()}")
            # 返回零损失以避免中断验证过程
            return torch.zeros(1, device=self.device), torch.zeros(3, device=self.device)
        
    
    def _convert_output_for_nms(self, y):
        """
        一个严谨的辅助函数，其唯一职责是将检测头 forward() 方法的输出 'y'，
        转换为一个可直接输入 custom_nms 的、格式统一的检测张量。

        Args:
            y (torch.Tensor): 来自检测头 forward() 的输出，
                            其形状通常为 (B, C, HW)，
                            其中 C = [x, y, w, h, cls_prob_0...N, h_pos]。

        Returns:
            torch.Tensor: 一个形状为 (B, N, C') 的张量，
                        其中 C' = [x, y, w, h, objectness_conf, cls_prob_0...N, h_pos]。
        """
        # 1. 输入验证与格式统一
        y = y.permute(0, 2, 1)

        # 2. 提取各个部分
        # y 的列格式: [x_center, y_center, w, h, cls_prob_0, ..., cls_prob_N, h_pos]
        box_xywh = y[:, :, :4]       # 边界框 (xywh)
        cls_probs = y[:, :, 4:-1]   # 所有类别的概率
        h_pos = y[:, :, -1:]        # h_pos

        # 3. 创建 NMS 所需的 "objectness confidence"
        # 在YOLOv8的推理流程中，通常使用每个框最可能类别的分数作为其置信度
        objectness_conf, _ = cls_probs.max(2, keepdim=True)

        # 4. 拼接成最终的、符合 custom_nms 输入格式的张量
        # 最终格式: [x, y, w, h, objectness_conf, cls_prob_0, ..., cls_prob_N, h_pos]
        detections = torch.cat([
            box_xywh,
            objectness_conf,
            cls_probs,
            h_pos
        ], dim=2)
        
        return detections
                
            

    def custom_nms(self,
                        prediction,
                         conf_thres=0.25,
                         iou_thres=0.45,
                         classes=None,
                         agnostic=False,
                         multi_label=False,
                         labels=(),
                         max_det=300):
        import torchvision  # 导入torchvision.ops模块
        """
        一个严谨、高效且功能完整的自定义NMS函数。
        它被设计用来处理一个特定的输入张量格式，并正确实现了所有过滤和注入逻辑。

        Args:
            prediction (torch.Tensor): 模型的预解码输出，形状为 (B, N, C)。
                                    C的列定义为: [x, y, w, h, objectness_conf, cls_score_0, ..., cls_score_N, h_pos]。
            conf_thres (float): 物体置信度阈值。
            iou_thres (float): IoU阈值。
            classes (list[int], optional): 需要保留的类别ID列表。
            agnostic (bool): 是否执行与类别无关的NMS。
            multi_label (bool): 每个框是否可以有多个标签。
            labels (tuple, optional): 需要注入的真实标签列表，用于特殊评估。
            max_det (int): 每张图片的最大检测数量。

        Returns:
            list[torch.Tensor]: NMS处理后的检测结果列表，每个张量形状为 (num_detections, 7)，
                                列为 [x1, y1, x2, y2, conf, cls, h_pos]。
        """
        # 1. 输入验证与参数准备
        device = prediction.device
        bs = prediction.shape[0]
        nc = prediction.shape[2] - 6  # 计算类别数量
        max_wh = 7680  # 最大坐标值，用于agnostic nms
        max_nms = 30000  # 进入NMS前的最大候选框数量

        # 2. 按置信度进行初步过滤
        # 我们只关心物体置信度大于阈值的候选框
        xc = prediction[:, :, 4] > conf_thres
        
        # 3. 主循环：逐个图像进行处理
        output = [torch.zeros((0, 7), device=device) for _ in range(bs)]
        for xi, x in enumerate(prediction):  # xi是图像索引, x是该图像的所有预测
            
            # 应用初始置信度过滤
            x = x[xc[xi]]

            # --- “标签注入”逻辑 ---
            # 如果为当前图像提供了labels，则将其作为“完美预测”注入
            if labels and len(labels) > xi and len(labels[xi]):
                lb = labels[xi]
                # 你的真实标签格式: (cls, x, y, w, h, cluster_id, h_rel)
                v = torch.zeros((len(lb), nc + 6), device=device)
                
                # 转换为NMS内部处理的格式: [xywh, obj_conf, cls_scores..., h_pos]
                v[:, :4] = lb[:, 1:5]  # xywh
                v[:, 4] = 1.0          # conf = 100%
                v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # one-hot编码类别
                v[:, -1] = lb[:, 6]   # h_rel作为h_pos
                
                # 将“完美预测”与模型的真实预测合并
                x = torch.cat((x, v), 0)

            # 如果此图像没有任何候选框（无论是来自模型还是注入），则跳到下一张
            if not x.shape[0]:
                print("没有任何候选框")
                continue

            box, objconf, cls, h_pos = x.split((4, 1, nc, 1), 1)
        
            if multi_label:
                # 多标签模式
                # i: box的索引, j: class的索引
                # 我们只关心那些类别分数也大于阈值的 (box, class) 组合
                i, j = (cls > conf_thres).nonzero(as_tuple=False).T
                # 组合成 [xywh, obj_conf, cls_idx, h_pos]
                x = torch.cat((box[i], x[i, 4:5], j.float().unsqueeze(1), h_pos[i]), 1)
            else:  # 单标签模式
                # 找到每个box最可能的类别及其分数
                conf, j = cls.max(1, keepdim=True)
                # 拼接成 [xywh, obj_conf, cls_idx, h_pos]
                x = torch.cat((box, x[:, 4:5], j.float(), h_pos), 1)
                # 过滤掉那些最高类别分数低于阈值的框 (这是标准库中的关键逻辑)
                x = x[conf.view(-1) > conf_thres]
                
            # 按 'classes' 参数进一步过滤
            if classes is not None and len(classes) > 0:
                x = x[torch.isin(x[:, 5], torch.tensor(classes, device=x.device))]
                
            # 如果再次过滤后没有任何框，则跳过
            n = x.shape[0]
            if not n:
                continue
            if n > max_nms: # 限制进入NMS的框数量
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]

            # --- NMS 核心操作 ---
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # 类别偏移
            boxes, scores = ops.xywh2xyxy(x[:, :4]) + c, x[:, 4]  # boxes (xyxy), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)

            # 限制最终检测数量
            if i.shape[0] > max_det:
                i = i[:max_det]

            # 准备最终输出
            final_dets = x[i]
            # 将box格式从xywh转换为最终的xyxy
            final_dets[:, :4] = ops.xywh2xyxy(final_dets[:, :4])
            # 输出格式：[x1, y1, x2, y2, conf, cls, h_pos]
            output[xi] = final_dets

        return output

    def postprocess(self, preds):
        """
        对模型输出进行后处理，转换为检测结果
        
        Args:
            preds: 模型输出，格式可能是list、tuple或dict
            
        Returns:
            processed: 处理后的检测结果列表
        """
        # 1. 从模型原始输出中提取预解码的 'y' 张量
        # 在验证阶段，preds 通常是 (y, other_data) 的元组
        y = preds[0] if isinstance(preds, (list, tuple)) else preds
        print(f"🔍 DEBUG: y complete sample: {y[0][0]}")

        # 2. 调用新的辅助函数，将 'y' 转换为 NMS 所需的格式
        detections = self._convert_output_for_nms(y)
        print(f"🔍 DEBUG: detections complete sample: {detections[0][0]}")
        # 如果转换失败，返回空结果
        if detections is None:
            batch_size = y.shape[0]
            return [torch.zeros(0, 7, device=self.device) for _ in range(batch_size)]
        try:
            conf_threshold = self.args.conf.item() if isinstance(self.args.conf, torch.Tensor) else self.args.conf
            iou_threshold = self.args.iou.item() if isinstance(self.args.iou, torch.Tensor) else self.args.iou
            if detections is not None:
                print("检测到预解码的检测结果，直接应用自定义NMS")
                # 应用自定义NMS处理
                processed = self.custom_nms(
                    detections,
                    conf_threshold,
                    iou_threshold,
                    agnostic=self.args.single_cls or self.args.agnostic_nms,
                    max_det=self.args.max_det,
                    labels=self.lb,
                    multi_label=True
                )
                
                # 打印调试信息
                print("\n" + "="*50)
                print("🎯 CHECKPOINT VALIDATE: 预解码检测结果处理")
                for i, p in enumerate(processed):
                    if len(p) > 0:
                        print(f"    processed[{i}] shape: {p.shape}")
                        print(f"    processed[{i}] format: xywh + conf + cls + h_pos")
                        print(f"    processed[{i}] 范围: [{p[:, :4].min().item():.4f}, {p[:, :4].max().item():.4f}]")
                    else:
                        print(f"    processed[{i}] 为空")
                print("="*50 + "\n")
                
                return processed
            
        except Exception as e:
            print(f"后处理出错: {e}")
            import traceback
            print(traceback.format_exc())
    
    def _prepare_batch(self, si, batch):
        """准备单个批次的数据"""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        imgsz = batch["img"].shape[2:]
        # 获取高度位置信息，如果存在的话
        h_pos = batch.get("h_rel", None)
        if h_pos is not None:
            h_pos = h_pos[idx]
        
        # 获取原始图像信息
        ori_shape = batch["ori_shape"][si]
        ratio_pad = batch["ratio_pad"][si]
        # 获取图像路径，如果存在的话
        im_file = batch.get("im_file", None)
        if im_file is not None and isinstance(im_file, (list, tuple)):
            im_file = im_file[si]
        
        # 创建目标字典
        target = {
            "imgsz": imgsz,  # 原始图像尺寸
            "cls": cls,             # 类别
            "bbox": bbox,           # 边界框
            "h_pos": h_pos,         # 高度位置
            "img_path": im_file,    # 图像路径
            "img_id": si,            # 图像ID
            "ratio_pad": ratio_pad,
            "ori_shape": ori_shape
        }
        
        return target

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """执行验证过程，在dataloader上运行推理并计算性能指标。"""
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        
        # 设置模型为tomato任务
        self.args.task = "tomato"
        
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            # force FP16 val during training
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            
            # 保存原始任务类型和处理器
            original_task = getattr(model, 'task', None)
            # 临时设置任务类型为tomato
            if hasattr(model, 'task'):
                model.task = 'tomato'
                
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            if str(self.args.model).endswith(".yaml") and model is None:
                LOGGER.warning("WARNING ⚠️ validating an untrained model YAML will result in 0 mAP.")
            callbacks.add_integration_callbacks(self)
            model = AutoBackend(
                weights=model or self.args.model,
                device=select_device(self.args.device, self.args.batch),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            # 设置任务类型为tomato
            if hasattr(model, 'task'):
                model.task = 'tomato'
                
            self.device = model.device  # update device
            self.args.half = model.fp16  # update half
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if engine:
                self.args.batch = model.batch_size
            elif not pt and not jit:
                self.args.batch = model.metadata.get("batch", 1)  # export.py models default to batch-size 1
                LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")

            # 加载数据集
            self.data = check_det_dataset(self.args.data)  # 使用检测数据集检查器
            LOGGER.info(f"Validating tomato detection model on {self.args.data}")

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
            if not pt:
                self.args.rect = False
            self.stride = model.stride  # used in get_dataloader() for padding
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

            model.eval()
            model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))  # warmup

        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)
            
            # Inference
            with dt[1]:
                # 模型推理 - 直接使用模型输出，不再执行格式转换
                preds = model(batch["img"], augment=augment)
                print(f"🔍 DEBUG: preds 类型: {type(preds)}")
                print(f"🔍 DEBUG: what is in preds: {type(preds[0])}")
            # Loss
            with dt[2]:
                if self.training:
                    # 使用安全损失计算方法
                    loss, loss_items = self.safe_compute_loss(model, batch, preds)
                    self.loss += loss_items
            # Postprocess
            with dt[3]:
                preds = self.postprocess(preds)
            
            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)
            
            self.run_callbacks("on_val_batch_end")
            
        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")
        
        # 恢复模型原始任务类型
        if self.training and original_task is not None and hasattr(model, 'task'):
            model.task = original_task
            
        if self.training:
            model.float()
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
        else:
            LOGGER.info(
                "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                    *tuple(self.speed.values())
                )
            )
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w") as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
        return stats

    def update_metrics(self, preds, batch):
        """更新评估指标。"""
        
        # 初始化stats字典，确保包含所有需要的键
        if not hasattr(self, 'stats') or self.stats is None:
            self.stats = {
                'tp': [], 
                'conf': [], 
                'pred_cls': [], 
                'target_cls': [],
                'target_img': []  # 确保有target_img键
            }
        
        # 确保stats字典中包含target_img键
        if 'target_img' not in self.stats:
            self.stats['target_img'] = []
        
        # 重写update_metrics方法，不再调用父类方法
        for si, pred in enumerate(preds):
            try:
                self.seen += 1
                npr = len(pred)
                stat = dict(
                    conf=torch.zeros(0, device=self.device),
                    pred_cls=torch.zeros(0, device=self.device),
                    tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
                )
                
                # 使用自己的_prepare_batch方法处理批次
                pbatch = self._prepare_batch(si, batch)
                cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
                nl = len(cls)
                stat["target_cls"] = cls
                stat["target_img"] = cls.unique()  # 确保设置target_img
                
                if npr == 0:
                    if nl:
                        # 使用append方法更新统计信息，确保使用PyTorch张量
                        self.stats['tp'].append(torch.zeros(0, self.niou, dtype=torch.bool, device=self.device))
                        self.stats['conf'].append(torch.zeros(0, device=self.device))
                        self.stats['pred_cls'].append(torch.zeros(0, device=self.device))
                        self.stats['target_cls'].append(stat["target_cls"])  # 已经是PyTorch张量
                        self.stats['target_img'].append(stat["target_img"])  # 添加target_img
                        
                        if self.args.plots:
                            # bbox已经是xyxy格式
                            self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                    continue

                # 处理预测结果
                if self.args.single_cls:
                    pred[:, 5] = 0
                    
                # 准备预测结果
                predn = self._prepare_pred( pred, pbatch)
                bbox_xyxy = ops.xywh2xyxy(bbox[:, :4].clone())
                stat["conf"] = predn[:, 4]
                stat["pred_cls"] = predn[:, 5]

                # =================== CHECKPOINT VALIDATE ===================
                print("\n" + "="*50)
                print("🎯 CHECKPOINT VALIDATE: IoU计算前的格式检查")
                print(f"    bbox (GT) 格式: xyxy (像素), 形状: {bbox_xyxy.shape}")
                print(f"    bbox value sample: {bbox[:5]}")
                print(f"    bbox 范围: [{bbox.min().item():.4f}, {bbox.max().item():.4f}]")
                print(f"    predn (Pred) 格式: xyxy[:4] + conf + cls + h_pos, 形状: {predn.shape}")
                print(f"    predn[:, :4] 范围: [{predn[:, :4].min().item():.4f}, {predn[:, :4].max().item():.4f}]")
                print(f"    predn[:, :4] sample: {predn[:, :4][:5]}")
                print("="*50 + "\n")
                # ====================================================
                
                # 计算IoU
                iou = box_iou(bbox_xyxy, predn[:, :4])  
                
                # 检查IoU计算结果
                if torch.isnan(iou).any():
                    print(f"警告: IoU计算结果包含NaN值，将替换为0")
                    iou = torch.nan_to_num(iou, nan=0.0)
                
                # =================== CHECKPOINT VALIDATE ===================
                print("\n" + "="*50)
                print("🎯 CHECKPOINT VALIDATE: IoU计算结果")
                print(f"    IoU矩阵形状: {iou.shape}")
                print(f"    IoU范围: [{iou.min().item():.4f}, {iou.max().item():.4f}]")
                print(f"    IoU包含NaN: {torch.isnan(iou).any().item()}")
                print("="*50 + "\n")
                # ====================================================
                
                # 根据IoU值分配预测框和真实框
                correct = np.zeros((npr, self.niou), dtype=bool)  # init
                if nl:
                    assigned_gt = torch.zeros((1, nl), dtype=torch.long, device=self.device) - 1
                    
                    # 找到每个预测框最匹配的真实框
                    max_iou, gt_idx = iou.max(0)  # best ious, indices
                    
                    # 根据IoU阈值判断是否为正样本
                    for k, t in enumerate(self.iouv):
                        correct_k = max_iou > t
                        # 记录每个IoU阈值下的正确预测
                        correct[:, k] = correct_k.cpu().numpy()
                    
                    # 确保correct是torch.Tensor而不是numpy.ndarray
                    correct_tensor = torch.from_numpy(correct).to(self.device)
                    
                    if self.args.save_json:
                        # 记录已分配的真实框
                        assigned_gt[:, gt_idx[max_iou > self.args.iou]] = torch.arange(max_iou.shape[0], device=self.device)[max_iou > self.args.iou]
                        
                        # 将预测结果添加到JSON字典
                        self.pred_to_json(predn, batch["im_file"][si], assigned_gt)
                
                # 更新统计信息
                # 使用append方法而不是索引赋值
                self.stats['tp'].append(correct_tensor)  # 使用torch.Tensor
                self.stats['conf'].append(predn[:, 4])
                self.stats['pred_cls'].append(predn[:, 5])
                self.stats['target_cls'].append(stat["target_cls"])
                self.stats['target_img'].append(stat["target_img"])  # 确保添加target_img

                # 处理h_pos特定指标
                if len(pred) > 0 and pred.shape[1] > 6:  # 确保有h_pos预测
                    h_pos_pred = pred[:, 6]
                    if nl:
                        try:
                            # 匹配检测和真实框
                            max_iou, max_idx = iou.max(0)
                            matched = max_iou > 0.5
                            
                            # =================== CHECKPOINT VALIDATE ===================
                            print("\n" + "="*50)
                            print("🎯 CHECKPOINT VALIDATE: H_pos Metrics")
                            print(f"    h_pos_pred: {h_pos_pred}")
                            print(f"    matched: {matched.sum().item()}/{len(matched)}")
                            print(f"    batch中是否有h_rel: {'h_rel' in batch}")
                            if 'h_rel' in batch:
                                h_rel_batch = batch['h_rel']
                                print(f"    h_rel_batch类型: {type(h_rel_batch)}")
                                if isinstance(h_rel_batch, list):
                                    print(f"    h_rel_batch是列表，长度: {len(h_rel_batch)}")
                                    if si < len(h_rel_batch):
                                        print(f"    h_rel_batch[{si}]类型: {type(h_rel_batch[si])}")
                                        print(f"    h_rel_batch[{si}]形状: {h_rel_batch[si].shape if hasattr(h_rel_batch[si], 'shape') else 'no shape'}")
                                else:
                                    print(f"    h_rel_batch形状: {h_rel_batch.shape}")
                                    print(f"    batch_idx中si={si}的数量: {(batch['batch_idx'] == si).sum().item()}")
                            print("="*50 + "\n")
                            # ====================================================
                            
                            # 检查batch中是否有h_rel字段
                            if matched.sum() > 0 and 'h_rel' in batch:
                                # 对于tomato任务，数据集collate_fn提供了h_rel字段
                                h_rel_batch = batch['h_rel']
                                if isinstance(h_rel_batch, list):
                                    if si < len(h_rel_batch):
                                        h_rel_true = h_rel_batch[si][max_idx[matched]]
                                        # 使用TomatoMetrics的update_h_pos_stats方法
                                        self.metrics.update_h_pos_stats(
                                            matched[matched], 
                                            pred[matched, 4], 
                                            h_pos_pred[matched], 
                                            h_rel_true
                                        )
                                else:
                                    # 如果是tensor，需要找到对应si的所有h_rel
                                    idx = batch["batch_idx"] == si
                                    if idx.any():
                                        h_idx = batch["batch_idx"] == si
                                        if len(h_rel_batch[h_idx]) > 0:
                                            h_rel_true = h_rel_batch[h_idx][max_idx[matched]]
                                            # 使用TomatoMetrics的update_h_pos_stats方法
                                            self.metrics.update_h_pos_stats(
                                                matched[matched], 
                                                pred[matched, 4], 
                                                h_pos_pred[matched], 
                                                h_rel_true
                                            )
                        except Exception as e:
                            LOGGER.warning(f"处理h_pos指标时出错: {e}")
                            import traceback
                            LOGGER.warning(f"错误详情: {traceback.format_exc()}")
            except Exception as e:
                LOGGER.warning(f"处理批次 {si} 时出错: {e}")
                import traceback
                LOGGER.warning(f"错误详情: {traceback.format_exc()}")
                continue
                
    def get_stats(self):
        """
        计算并返回所有验证指标。
        这个版本正确地将番茄特定指标与父类中的标准指标合并。
        """
        try:
            # 确保stats字典包含所有必要的键
            if 'target_img' not in self.stats:
                LOGGER.warning("警告: stats字典中缺少'target_img'键，使用空列表")
                self.stats['target_img'] = []
                
            # 检查是否有任何空列表，如果有，添加占位符避免错误
            for k in ['tp', 'conf', 'pred_cls', 'target_cls', 'target_img']:
                if len(self.stats[k]) == 0:
                    LOGGER.warning(f"警告: stats['{k}']为空，添加占位符")
                    self.stats[k] = [torch.zeros(1, device=self.device)]

            # 1. 将列表合并为张量
            stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}
            
            # 2. 计算每个类别的统计信息
            self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
            
            # 3. 计算每张图像的目标数量
            if len(stats["target_img"]) > 0:
                self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)
            else:
                self.nt_per_image = np.zeros(self.nc)
                
            # 4. 从stats中移除target_img，因为后续处理不需要
            stats.pop("target_img", None)
            
            # 5. 处理tp、conf等计算mAP
            if len(stats) and stats["tp"].any():
                self.metrics.process(**stats)
                
            # 6. 计算番茄特定指标
            self.metrics.finalize_h_pos_metrics()
            h_mae = self.metrics.h_pos_stats.get('h_mae', 0.0)
            rank_acc = self.metrics.h_pos_stats.get('rank_acc', 0.0)
            
            # 7. 更新结果字典
            results = self.metrics.results_dict
            results['metrics/h_mae'] = h_mae
            results['metrics/rank_acc'] = rank_acc
            
            return results
            
        except Exception as e:
            LOGGER.warning(f"计算统计信息时出错: {e}")
            import traceback
            LOGGER.warning(f"错误详情: {traceback.format_exc()}")
            # 返回最小化的结果字典，避免完全失败
            return {
                'metrics/precision': 0.0,
                'metrics/recall': 0.0,
                'metrics/mAP50': 0.0,
                'metrics/mAP50-95': 0.0,
                'metrics/h_mae': 0.0,
                'metrics/rank_acc': 0.0
            }
        
    def finalize_metrics(self, *args, **kwargs):
        """完成指标计算，添加h_pos评估和排序指标。"""
        # 调用父类方法处理常规检测指标
        super().finalize_metrics(*args, **kwargs)
        
        # 使用TomatoMetrics的finalize_h_pos_metrics方法
        self.metrics.finalize_h_pos_metrics()
        
        # 添加h_pos指标到结果中
        h_mae = self.metrics.h_pos_stats.get('h_mae', 0.0)
        rank_acc = self.metrics.h_pos_stats.get('rank_acc', 0.0)
        
        # 创建一个新的字典来存储结果，而不是尝试修改父类方法的返回值
        metrics = {}
        metrics['metrics/h_mae'] = h_mae
        metrics['metrics/rank_acc'] = rank_acc
        
        LOGGER.info(f"H-pos MAE: {h_mae:.4f}")
        LOGGER.info(f"H-pos Ranking Accuracy: {rank_acc:.4f}")
        
        return metrics
        
    def get_desc(self):
        """返回格式化的字符串，总结YOLO模型的类指标。"""
        return ("%22s" + "%11s" * 7) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95", "H-MAE") 
