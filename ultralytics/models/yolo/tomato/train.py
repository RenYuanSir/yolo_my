# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import re
import logging
from ultralytics.utils.torch_utils import model_info

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import TomatoDetectionModel
from ultralytics.utils import LOGGER
from ultralytics.models import yolo
from copy import copy
from ultralytics.utils.loss import TomatoDetectWithRankLoss

LOGGER = logging.getLogger(__name__)

# 添加解析损失函数字符串参数的函数
def parse_loss_args(loss_str):
    """
    从损失函数字符串中解析参数
    
    Args:
        loss_str (str): 损失函数字符串，如 "TomatoDetectWithRankLoss(lambda_rank=5)"
        
    Returns:
        dict: 解析出的参数字典，如 {"lambda_rank": 5.0}
    """
    if not loss_str:
        return {}
    
    # 提取括号内的参数部分
    match = re.search(r'\((.*?)\)$', loss_str)
    if not match:
        return {}
    
    params_str = match.group(1)
    params = {}
    
    # 解析参数
    for param in params_str.split(','):
        if '=' not in param:
            continue
        key, value = param.strip().split('=', 1)
        key = key.strip()
        value = value.strip()
        
        # 尝试将值转换为数字
        try:
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            # 如果不是数字，保留为字符串
            pass
            
        params[key] = value
        
    return params

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

    def get_validator(self):
            """Returns a TomatoValidator instance for validation."""
            # This overrides the parent class and ensures your custom validator is used.
            return yolo.tomato.TomatoValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def set_rank_params(self, lambda_rank=0.2, margin=0.1, tal_topk=10):
        """
        设置排序损失参数
        
        Args:
            lambda_rank (float): 排序损失权重
            margin (float): 排序边界
            tal_topk (int): 任务对齐分配器的topk数量
        """
        # 从args中获取loss字符串参数
        loss_str = getattr(self.args, 'loss', None)
        if loss_str and 'TomatoDetectWithRankLoss' in loss_str:
            # 解析损失函数参数
            params = parse_loss_args(loss_str)
            
            # 如果在loss字符串中指定了lambda_rank，则使用该值
            if 'lambda_rank' in params:
                lambda_rank = params['lambda_rank']
                LOGGER.info(f"从loss参数中解析得到lambda_rank={lambda_rank}")
            
            # 如果在loss字符串中指定了margin，则使用该值
            if 'margin' in params:
                margin = params['margin']
                LOGGER.info(f"从loss参数中解析得到margin={margin}")
                
            # 如果在loss字符串中指定了tal_topk，则使用该值
            if 'tal_topk' in params:
                tal_topk = params['tal_topk']
                LOGGER.info(f"从loss参数中解析得到tal_topk={tal_topk}")
        
        # 设置模型参数
            self.model.set_rank_params(lambda_rank, margin, tal_topk)
            LOGGER.info(f"设置排序损失参数: lambda_rank={lambda_rank}, margin={margin}, tal_topk={tal_topk}")
            
    @property
    def loss_names(self):
        """返回损失组件名称。"""
        return 'box_loss', 'cls_loss', 'dfl_loss', 'rank_loss'

    @loss_names.setter
    def loss_names(self, value):
        """防止BaseTrainer中的AttributeError。"""
        pass 