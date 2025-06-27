# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import TomatoDetectionModel
from ultralytics.utils import LOGGER
from ultralytics.utils.loss import TomatoDetectWithRankLoss


class TomatoTrainer(DetectionTrainer):
    """
    TomatoTrainer 类用于训练针对番茄成熟度等级检测和串识别的番茄检测模型。
    
    继承自DetectionTrainer，添加了特定于番茄检测的功能，如排序损失和h_pos预测。
    
    Example:
        ```python
        from ultralytics.models.yolo.tomato import TomatoTrainer

        args = dict(model="yolov12-tomato.pt", data="tomato.yaml", epochs=50)
        trainer = TomatoTrainer(overrides=args)
        trainer.train()
        ```
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        """返回一个番茄检测模型。"""
        model = TomatoDetectionModel(cfg, nc=self.data["nc"], verbose=verbose)
        if weights:
            model.load(weights)
        return model

    def get_criterion(self):
        """获取训练的损失函数，默认使用TomatoDetectWithRankLoss。"""
        if self.args.loss and 'TomatoDetectWithRankLoss' not in self.args.loss:
            return super().get_criterion()
        
        # 使用番茄检测特定的损失函数
        LOGGER.info("使用番茄检测特定的损失函数：TomatoDetectWithRankLoss")
        return TomatoDetectWithRankLoss(self.model)

    def set_rank_params(self, lambda_rank=0.2, margin=0.1, tal_topk=10):
        """
        设置排序损失相关参数
        
        Args:
            lambda_rank (float): 排序损失权重
            margin (float): 排序边界
            tal_topk (int): 任务对齐学习中的top-k
        """
        if hasattr(self.model, "set_rank_params"):
            self.model.set_rank_params(lambda_rank, margin, tal_topk)
            LOGGER.info(f"设置排序损失参数: lambda_rank={lambda_rank}, margin={margin}, tal_topk={tal_topk}")
        else:
            LOGGER.warning("模型不支持排序参数设置，可能不是番茄检测模型")
            
    @property
    def loss_names(self):
        """返回损失组件名称。"""
        return 'box_loss', 'cls_loss', 'dfl_loss', 'rank_loss'

    @loss_names.setter
    def loss_names(self, value):
        """防止BaseTrainer中的AttributeError。"""
        pass 