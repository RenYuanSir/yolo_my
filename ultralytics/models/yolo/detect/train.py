# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
import random
from copy import copy

import numpy as np
import torch.nn as nn

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first
from ultralytics.utils.loss import v8DetectionLoss


class DetectionTrainer(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionTrainer

        args = dict(model="yolo11n.pt", data="coco8.yaml", epochs=3)
        trainer = DetectionTrainer(overrides=args)
        trainer.train()
        ```
    """

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Construct and return dataloader."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        """Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    @property
    def loss_names(self):
        """Returns the names of the loss components."""
        if self.args.loss and 'TomatoDetectWithRankLoss' in self.args.loss:
            return 'box_loss', 'cls_loss', 'dfl_loss', 'rank_loss'
        else:
            return 'box_loss', 'cls_loss', 'dfl_loss'

    @loss_names.setter
    def loss_names(self, value):
        """
        Setter for loss_names to prevent AttributeError from BaseTrainer.
        This property is dynamically calculated, so the setter does nothing.
        """
        pass

    def progress_string(self):
        """Returns a formatted string of training progress with epoch, memory, loss, instances, and image size."""
        # This method is overridden from BaseTrainer to handle custom loss names
        loss_names_count = len(self.loss_names)
        format_str = "%11s" * 2 + "%11s" * loss_names_count + "%11s" * 2
        
        # 首先创建占位字符串
        placeholder = (
            "Epoch",
            "GPU_mem",
            *self.loss_names,  # 损失名称作为列标题
            "Instances",
            "Size",
        )
        
        return format_str % placeholder

    def plot_training_samples(self, batch, ni):
        """Plot train image batch with labels."""
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch.get("im_file", ["" for _ in range(len(batch["img"]))]), # 使用get方法，如果没有im_file则提供默认值
            fname=self.save_dir / f"train_batch{ni}.jpg",
        )

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        try:
            # 过滤出字典类型的标签并确保它们包含必要的键
            valid_labels = []
            for lb in self.train_loader.dataset.labels:
                if isinstance(lb, dict) and "bboxes" in lb and "cls" in lb:
                    valid_labels.append(lb)
                elif not isinstance(lb, dict):
                    LOGGER.warning(f"跳过非字典类型标签: {type(lb)}")
                else:
                    LOGGER.warning(f"跳过缺少必要键的标签: {lb.keys() if isinstance(lb, dict) else type(lb)}")
            
            if not valid_labels:
                LOGGER.warning("没有有效的标签用于绘图，跳过plot_training_labels")
                return
                
            boxes = np.concatenate([lb["bboxes"] for lb in valid_labels], 0)
            cls = np.concatenate([lb["cls"] for lb in valid_labels], 0)
            plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)
        except Exception as e:
            LOGGER.warning(f"绘制训练标签时出错：{e}，跳过此步骤")
            # 不让错误中断训练过程

    def auto_batch(self):
        """Get batch size by calculating memory occupation of model."""
        train_dataset = self.build_dataset(self.trainset, mode="train", batch=16)
        # 4 for mosaic augmentation
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4
        return super().auto_batch(max_num_obj)

    def get_criterion(self):
        """Returns the loss function for the detection task, supporting custom loss functions."""
        if self.args.loss:
            # Custom loss function specified
            import re
            from ultralytics.utils import loss as loss_module

            loss_str = self.args.loss
            match = re.match(r"(\w+)\((.*)\)", loss_str)
            if not match:
                raise ValueError(f"Invalid format for custom loss: {loss_str}. Expected 'ClassName(param1=value1,...)'.")
            
            class_name = match.group(1)
            params_str = match.group(2)

            # Check if the loss class exists in the loss module
            if not hasattr(loss_module, class_name):
                raise ValueError(f"Custom loss class '{class_name}' not found in 'ultralytics.utils.loss'.")
            
            loss_class = getattr(loss_module, class_name)
            
            # Safely parse parameters
            try:
                # Use eval to create a dictionary from the parameters string
                params = eval(f"dict({params_str})")
            except Exception as e:
                raise ValueError(f"Error parsing parameters for custom loss '{class_name}': {e}")

            return loss_class(self.model, **params)
        else:
            # Default loss function
            return v8DetectionLoss(self.model)
