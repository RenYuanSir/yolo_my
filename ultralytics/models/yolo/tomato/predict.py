# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import ops
from ultralytics.utils.plotting import colors, save_one_box


class TomatoPredictor(DetectionPredictor):
    """
    TomatoPredictor类用于番茄检测模型的推理，继承自DetectionPredictor。
    
    拓展功能包括处理h_pos预测和可视化番茄成熟度等级。
    
    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.tomato import TomatoPredictor

        args = dict(model="yolov12-tomato.pt", source=ASSETS)
        predictor = TomatoPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def postprocess(self, preds, img, orig_imgs):
        """对预测结果进行后处理，并返回Results对象列表。"""
        # 检查是否包含h_pos预测
        if isinstance(preds, dict) and 'h_pos' in preds:
            h_pos = preds['h_pos']  # 保存h_pos预测
            preds = preds['features']  # 获取边界框预测
            
            # 对边界框预测进行NMS
            preds = ops.non_max_suppression(
                preds,
                self.args.conf,
                self.args.iou,
                agnostic=self.args.agnostic_nms,
                max_det=self.args.max_det,
                classes=self.args.classes,
            )
            
            if not isinstance(orig_imgs, list):  # 如果输入图像是torch.Tensor而不是列表
                orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
            
            results = []
            for pred, h, orig_img, img_path in zip(preds, h_pos, orig_imgs, self.batch[0]):
                # 缩放边界框到原始图像尺寸
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                
                # 处理h_pos预测
                if len(pred):
                    h = h.flatten(2).permute(0, 2, 1)  # (bs, n_points, 1)
                    h_idx = torch.zeros(len(pred), dtype=torch.long, device=h.device)
                    h_values = h[0][h_idx].squeeze(-1)  # 提取对应的h_pos值
                    
                    # 创建Results对象
                    result = Results(orig_img, path=img_path, names=self.model.names, boxes=pred)
                    # 添加h_pos信息
                    result.h_pos = h_values
                else:
                    # 空结果
                    result = Results(orig_img, path=img_path, names=self.model.names, boxes=pred)
                    result.h_pos = torch.tensor([])
                
                results.append(result)
            
            return results
        else:
            # 常规后处理
            return super().postprocess(preds, img, orig_imgs)
    
    def write_results(self, idx, results, batch):
        """将结果写入文件并可视化。"""
        # 获取标准结果输出
        super().write_results(idx, results, batch)
        
        # 如果结果包含h_pos信息，添加到输出
        p, im, im0s = batch
        result = results[idx]
        
        if hasattr(result, 'h_pos') and len(result.h_pos) > 0:
            # 将h_pos信息添加到标签
            boxes = result.boxes
            if len(boxes):
                cls = boxes.cls.cpu().tolist()
                conf = boxes.conf.cpu().tolist()
                h_pos = result.h_pos.cpu().tolist()
                
                # 修改标签以包含h_pos信息
                for i, (c, conf, h) in enumerate(zip(cls, conf, h_pos)):
                    # 获取类名和置信度
                    label = f"{result.names[int(c)]} {conf:.2f} h:{h:.2f}"
                    
                    # 如果启用了保存带注释的图像
                    if self.args.save or self.args.save_txt or self.args.show:
                        # 获取边界框
                        box = boxes.xyxy[i].cpu().tolist()
                        
                        # 获取颜色，可以根据h_pos值设置不同的颜色
                        color = colors(int(c), True)  # 直接使用类别颜色
                        
                        # 在图像上绘制h_pos值
                        if self.args.save or self.args.show:
                            # 这一行已经在原始方法中完成
                            pass  # 避免重复绘制
                
                # 如果需要保存带注释的图像
                if self.args.save_crop:
                    # 保存裁剪的检测结果
                    for i, (box, h) in enumerate(zip(boxes.xyxy, result.h_pos)):
                        # 获取类别索引
                        c = int(boxes.cls[i])
                        # 构建文件名，包含h_pos信息
                        crop_name = f"{self.files[idx].stem}_h{h.item():.2f}_{i}"
                        save_one_box(
                            box.cpu(),
                            im0s,
                            file=self.save_dir / "crops" / result.names[c] / f"{crop_name}.jpg",
                            BGR=True,
                        ) 