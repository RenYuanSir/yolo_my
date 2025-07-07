# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import json
from collections import defaultdict
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import ConcatDataset

from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM, colorstr
from ultralytics.utils.ops import resample_segments, xywhn2xyxy
from ultralytics.utils.torch_utils import TORCHVISION_0_18

from .augment import (
    Compose,
    Format,
    Instances,
    LetterBox,
    RandomLoadText,
    classify_augmentations,
    classify_transforms,
    v8_transforms,
)
from .base import BaseDataset
from .utils import (
    HELP_URL,
    LOGGER,
    IMG_FORMATS,
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    save_dataset_cache_file,
    verify_image,
    verify_image_label,
    exif_size,
)

# Ultralytics dataset *.cache version, >= 1.0.0 for YOLOv8
DATASET_CACHE_VERSION = "1.0.3"

# 添加缺少的函数
def box_iou_xywh(box1, box2):
    """
    计算两组框的IoU，输入格式为xywh
    
    Args:
        box1 (np.ndarray): 第一组框，形状为(n, 4)，格式为xywh
        box2 (np.ndarray): 第二组框，形状为(m, 4)，格式为xywh
        
    Returns:
        np.ndarray: IoU矩阵，形状为(n, m)
    """
    # 转换为xyxy格式
    b1_x1, b1_y1 = box1[:, 0] - box1[:, 2] / 2, box1[:, 1] - box1[:, 3] / 2
    b1_x2, b1_y2 = box1[:, 0] + box1[:, 2] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_y1 = box2[:, 0] - box2[:, 2] / 2, box2[:, 1] - box2[:, 3] / 2
    b2_x2, b2_y2 = box2[:, 0] + box2[:, 2] / 2, box2[:, 1] + box2[:, 3] / 2
    
    # 计算交集区域
    inter_rect_x1 = np.maximum(b1_x1[:, None], b2_x1)
    inter_rect_y1 = np.maximum(b1_y1[:, None], b2_y1)
    inter_rect_x2 = np.minimum(b1_x2[:, None], b2_x2)
    inter_rect_y2 = np.minimum(b1_y2[:, None], b2_y2)
    
    # 计算交集面积
    inter_area = np.clip(inter_rect_x2 - inter_rect_x1, 0, None) * \
                 np.clip(inter_rect_y2 - inter_rect_y1, 0, None)
    
    # 计算各自面积
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    # 计算IoU
    iou = inter_area / (b1_area[:, None] + b2_area - inter_area + 1e-16)
    
    return iou

def box_iou(box1, box2):
    """
    计算两组框的IoU，输入格式为xyxy
    
    Args:
        box1 (np.ndarray): 第一组框，形状为(n, 4)，格式为xyxy
        box2 (np.ndarray): 第二组框，形状为(m, 4)，格式为xyxy
        
    Returns:
        np.ndarray: IoU矩阵，形状为(n, m)
    """
    # 计算交集区域
    inter_rect_x1 = np.maximum(box1[:, 0][:, None], box2[:, 0])
    inter_rect_y1 = np.maximum(box1[:, 1][:, None], box2[:, 1])
    inter_rect_x2 = np.minimum(box1[:, 2][:, None], box2[:, 2])
    inter_rect_y2 = np.minimum(box1[:, 3][:, None], box2[:, 3])
    
    # 计算交集面积
    inter_area = np.clip(inter_rect_x2 - inter_rect_x1, 0, None) * \
                 np.clip(inter_rect_y2 - inter_rect_y1, 0, None)
    
    # 计算各自面积
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    # 计算IoU
    iou = inter_area / (box1_area[:, None] + box2_area - inter_area + 1e-16)
    
    return iou

class YOLODataset(BaseDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        task (str): An explicit arg to point current task, Defaults to 'detect'.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """

    def __init__(self, *args, data=None, task="detect", **kwargs):
        """Initializes the YOLODataset with optional configurations for segments and keypoints."""
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        super().__init__(*args, **kwargs)

    def cache_labels(self, path=Path("./labels.cache")):
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): Path where to save the cache file. Default is Path("./labels.cache").

        Returns:
            (dict): labels.
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],  # n, 1
                            "bboxes": lb[:, 1:],  # n, 4
                            "segments": segments,
                            "keypoints": keypoint,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            LOGGER.warning(f"WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}")
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"WARNING ⚠️ No labels found in {cache_path}, training may not work correctly. {HELP_URL}")
        return labels

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms

    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        """
        Custom your label format here.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # make sure segments interpolate correctly if original length is greater than segment_resamples
            max_len = max(len(s) for s in segments)
            segment_resamples = (max_len + 1) if segment_resamples < max_len else segment_resamples
            # list[np.array(segment_resamples, 2)] * num_samples
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            if k in {"cluster_ids", "h_rel"}:
                # 确保所有元素都是PyTorch张量
                tensors = []
                for v in value:
                    if isinstance(v, np.ndarray):
                        tensors.append(torch.from_numpy(v))
                    else:
                        tensors.append(v)
                value = torch.cat(tensors, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch


class TomatoYOLODataset(YOLODataset):
    """
    番茄检测专用数据集类，扩展了YOLODataset以支持番茄串ID和相对高度标签。
    
    该类专为番茄检测任务设计，扩展了标准YOLO格式，每个标签行包含：
    [class, x, y, w, h, cluster_ids, h_rel]
    其中:
    - class: 类别ID (0-5，包括不同成熟度的番茄和番茄串)
    - x, y, w, h: 标准YOLO格式的边界框坐标（归一化）
    - cluster_ids: 番茄串ID，相同ID的番茄属于同一串
    - h_rel: 相对高度，表示番茄在串中的相对位置（0-1，0表示最高位置）
    """
    def __init__(self, *args, data=None, task="detect", **kwargs):
        """
        初始化番茄检测数据集
        
        Args:
            data: 数据集配置字典
            *args, **kwargs: 传递给YOLODataset的其他参数
        """
        # 提取args参数（如果在kwargs中）
        train_args = kwargs.pop('args', None)
        
        # 确保data字典存在
        if data is None:
            data = {"names": ["tomato"], "nc": 1}  # 提供默认的数据配置
            LOGGER.warning(f"警告: data参数为None，使用默认配置。")
        elif not isinstance(data, dict):
            # 如果data不是字典类型，创建一个默认字典
            LOGGER.warning(f"警告: data参数不是字典类型: {type(data)}. 使用默认配置。")
            data = {"names": ["tomato"], "nc": 1}
        
        # 确保data字典中包含必要的键
        if "names" not in data:
            data["names"] = ["tomato"]
        if "nc" not in data:
            data["nc"] = len(data["names"])
        
        # 设置番茄数据集特有的标志
        self.has_cluster_ids = data.get("has_cluster_ids", True)
        self.has_h_rel = data.get("has_h_rel", True)
        
        # 确保data字典中包含这些标志
        data["has_cluster_ids"] = self.has_cluster_ids
        data["has_h_rel"] = self.has_h_rel
        
        # 检查是否使用TomatoDetectWithRankLoss
        if train_args and hasattr(train_args, 'loss') and 'TomatoDetectWithRankLoss' in train_args.loss:
            LOGGER.info(f"检测到使用TomatoDetectWithRankLoss，启用cluster_ids和h_rel支持")
            self.has_cluster_ids = True
            self.has_h_rel = True
            data["has_cluster_ids"] = True
        data["has_h_rel"] = True
        
        # 设置YOLODataset所需的属性
        self.use_segments = False
        self.use_keypoints = False
        self.use_obb = False
        self.data = data
        
        # 直接调用BaseDataset的__init__方法，跳过YOLODataset的__init__方法
        BaseDataset.__init__(self, *args, data=data, **kwargs)
        
        # 输出初始化信息
        LOGGER.info(f"初始化番茄数据集: has_cluster_ids={self.has_cluster_ids}, has_h_rel={self.has_h_rel}")
        
    def cache_labels(self, path=Path("./labels.cache")):
        """
        缓存数据集标签，检查图像并读取形状。
        重写YOLODataset的cache_labels方法，确保正确处理番茄特有的标签格式。
        
        Args:
            path (Path): 保存缓存文件的路径。默认为Path("./labels.cache")。
            
        Returns:
            (dict): 标签。
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        
        LOGGER.info(f"从标签文件中读取扩展信息")
        
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(nkpt),
                    repeat(ndim),
                    repeat(self.use_keypoints),
                    repeat(self.use_segments),
                    repeat(self.use_obb),
                    repeat(self.has_cluster_ids),  # 传递has_cluster_ids参数
                    repeat(self.has_h_rel),  # 传递has_h_rel参数
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            valid_label_count = 0  # 跟踪有效标签数量
            
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f  # 这个值可能不准确，因为verify_image_label在有标签时不会设置nf_f=1
                ne += ne_f
                nc += nc_f
                
                if im_file:
                    has_valid_labels = len(lb) > 0
                    
                    # 创建标签字典
                    label_dict = {
                        "im_file": im_file,
                        "shape": shape,
                        "cls": lb[:, 0:1],  # n, 1
                        "bboxes": lb[:, 1:5],  # n, 4
                        "segments": segments,
                        "keypoints": keypoint,
                        "normalized": True,
                        "bbox_format": "xywh",
                    }
                    
                    # 添加cluster_ids和h_rel字段
                    if self.has_cluster_ids and lb.shape[1] > 5:
                        label_dict["cluster_ids"] = lb[:, 5:6]
                    if self.has_h_rel and lb.shape[1] > 6:
                        label_dict["h_rel"] = lb[:, 6:7]
                    
                    x["labels"].append(label_dict)
                    
                    # 如果有有效标签，增加计数
                    if has_valid_labels:
                        valid_label_count += 1
                
                if msg:
                    msgs.append(msg)
                
                # 更新进度条描述，显示有标签的图像数量和无标签的图像数量
                pbar.desc = f"{desc} {valid_label_count} labeled images, {nm + ne} backgrounds, {nc} corrupt"
            
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        
        # 确保有标签的图像数量正确
        labeled_images = sum(1 for lb in x["labels"] if len(lb["cls"]) > 0)
        if labeled_images == 0:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ No labeled images found in {path}. {HELP_URL}")
        
        # 更新nf为实际找到的有标签图像数量
        nf = labeled_images
        
        # 更新结果
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        
        # 添加额外的日志，确保用户了解实际情况
        LOGGER.info(f"缓存统计: 总图像数={len(self.im_files)}, 有标签图像数={nf}, 无标签图像数={nm + ne}, 损坏图像数={nc}")
        
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x
    
    def get_labels(self):
        """
        获取数据集标签，扩展以支持cluster_ids和h_rel
        
        Returns:
            dict: 标签字典
        """
        # 首先生成标签文件路径
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        
        # 尝试直接加载缓存文件
        try:
            cache, exists = load_dataset_cache_file(cache_path), True
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash(self.label_files + self.im_files)
        except (FileNotFoundError, AssertionError, AttributeError):
            # 如果缓存文件不存在或无效，则重新生成缓存
            LOGGER.info(f"缓存文件不存在或无效，重新生成缓存: {cache_path}")
            cache, exists = self.cache_labels(cache_path), False
        
        # 显示缓存信息
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            # 计算有标签的图像数量
            labeled_images = nf - (nm + ne)
            
            # 更新描述信息，明确显示有标签的图像数量和无标签的图像数量
            d = f"Scanning {cache_path}... {labeled_images} labeled images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))
            
            # 添加额外的日志，确保用户了解实际情况
            LOGGER.info(f"数据集统计: 总图像数={n}, 有标签图像数={labeled_images}, 无标签图像数={nm + ne}, 损坏图像数={nc}")
        
        # 读取缓存
        [cache.pop(k) for k in ("hash", "version", "msgs")]
        labels = cache["labels"]
        if not labels:
            LOGGER.warning(f"WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}")
        self.im_files = [lb["im_file"] for lb in labels]
        
        # 检查是否包含cluster_ids和h_rel字段
        has_cluster_ids = any("cluster_ids" in lb for lb in labels)
        has_h_rel = any("h_rel" in lb for lb in labels)
        
        if has_cluster_ids:
            LOGGER.info(f"数据集包含cluster_ids字段")
        if has_h_rel:
            LOGGER.info(f"数据集包含h_rel字段")
        
        return labels

    def update_labels_info(self, label):
        """
        更新标签信息，添加cluster_ids和h_rel支持，并确保坐标归一化
        
        Args:
            label: 标签字典
            
        Returns:
            dict: 更新后的标签字典
        """
        # 添加调试日志
        LOGGER.debug(f"TomatoYOLODataset.update_labels_info: 原始标签字段: {list(label.keys())}")
        
        # 首先检查边界框是否存在且有效
        if "bboxes" not in label or not isinstance(label["bboxes"], (np.ndarray, torch.Tensor)) or label["bboxes"].shape[0] == 0:
            # 如果没有边界框或边界框为空，调用父类方法
            label = super().update_labels_info(label)
            
            # 确保有必要的字段
            if self.has_cluster_ids and "cluster_ids" not in label:
                label["cluster_ids"] = np.zeros((0, 1), dtype=np.float32)
            if self.has_h_rel and "h_rel" not in label:
                label["h_rel"] = np.zeros((0, 1), dtype=np.float32)
                
            return label
            
        # 记录原始边界框形状
        bboxes = label["bboxes"]
        original_shape = bboxes.shape
        LOGGER.debug(f"TomatoYOLODataset.update_labels_info: 原始边界框形状: {original_shape}")
        
        # 检查坐标是否需要归一化（如果值大于1.0）
        if isinstance(bboxes, np.ndarray) and np.any(bboxes > 1.0):
            # 获取图像尺寸
            img_h, img_w = None, None
            if "img_hw" in label:
                img_h, img_w = label["img_hw"]
            elif "img" in label:
                img_h, img_w = label["img"].shape[:2] if isinstance(label["img"], np.ndarray) else label["img"].shape[-2:]
            elif "shape" in label:
                img_h, img_w = label["shape"]
            elif "ori_shape" in label:
                img_h, img_w = label["ori_shape"]
                
            # 如果能获取到图像尺寸，则进行归一化
            if img_h is not None and img_w is not None:
                is_xyxy = (bboxes[:, 2:4] > bboxes[:, 0:2]).all()  # 检查是否为xyxy格式
                
                if is_xyxy:
                    # XYXY格式边界框归一化
                    LOGGER.debug(f"TomatoYOLODataset.update_labels_info: 检测到XYXY格式边界框，进行归一化 (/{img_w}, /{img_h})")
                    bboxes[:, [0, 2]] /= img_w  # 归一化x坐标
                    bboxes[:, [1, 3]] /= img_h  # 归一化y坐标
                    
                    # 转换为中心点格式(XYWH)
                    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
                    cx = (x1 + x2) / 2  # 中心点x坐标
                    cy = (y1 + y2) / 2  # 中心点y坐标
                    w = x2 - x1         # 宽度
                    h = y2 - y1         # 高度

                    # 检查并修正异常宽度和高度
                    w_clamped = np.clip(w, 1e-6, None)
                    h_clamped = np.clip(h, 1e-6, None)
                    if np.any(w != w_clamped) or np.any(h != h_clamped):
                        LOGGER.warning(
                            "TomatoYOLODataset.update_labels_info: Detected non-positive width/height when converting from XYXY; clamped to epsilon"
                        )
                    w, h = w_clamped, h_clamped

                    # 重组并更新边界框
                    bboxes = np.stack([cx, cy, w, h], axis=1)
                    LOGGER.debug(f"TomatoYOLODataset.update_labels_info: 转换为XYWH格式: {bboxes.shape}")
                else:
                    # XYWH格式但需要归一化
                    LOGGER.debug(f"TomatoYOLODataset.update_labels_info: 检测到未归一化的XYWH格式边界框，进行归一化")
                    bboxes[:, 0] /= img_w  # 归一化中心点x
                    bboxes[:, 1] /= img_h  # 归一化中心点y
                    bboxes[:, 2] /= img_w  # 归一化宽度
                    bboxes[:, 3] /= img_h  # 归一化高度
                    w_clamped = np.clip(bboxes[:, 2], 1e-6, None)
                    h_clamped = np.clip(bboxes[:, 3], 1e-6, None)
                    if np.any(bboxes[:, 2] != w_clamped) or np.any(bboxes[:, 3] != h_clamped):
                        LOGGER.warning(
                            "TomatoYOLODataset.update_labels_info: Detected non-positive width/height during normalization; clamped to epsilon"
                        )
                    bboxes[:, 2] = w_clamped
                    bboxes[:, 3] = h_clamped
                
                # 更新标签中的边界框
                label["bboxes"] = bboxes
                label["normalized"] = True
                
                LOGGER.debug(f"TomatoYOLODataset.update_labels_info: 归一化后的边界框范围: x=[{bboxes[:, 0].min():.4f}, {bboxes[:, 0].max():.4f}], y=[{bboxes[:, 1].min():.4f}, {bboxes[:, 1].max():.4f}]")
                LOGGER.debug(f"TomatoYOLODataset.update_labels_info: 归一化后的边界框范围: w=[{bboxes[:, 2].min():.4f}, {bboxes[:, 2].max():.4f}], h=[{bboxes[:, 3].min():.4f}, {bboxes[:, 3].max():.4f}]")
        
        # 检查边界框是否包含扩展信息 (额外的列)
        bbox_shape = label["bboxes"].shape
        if bbox_shape[1] > 4:
            LOGGER.debug(f"TomatoYOLODataset.update_labels_info: 发现扩展边界框，列数={bbox_shape[1]}")
            
            # 提取标准的4列边界框
            bboxes_standard = label["bboxes"][:, :4]
            
            # 处理cluster_ids (第5列)
            if bbox_shape[1] >= 5 and "cluster_ids" not in label and self.has_cluster_ids:
                if isinstance(label["bboxes"], np.ndarray):
                    label["cluster_ids"] = label["bboxes"][:, 4:5]
                else:  # torch.Tensor
                    label["cluster_ids"] = label["bboxes"][:, 4:5].clone()  # 使用clone防止共享内存
                LOGGER.debug(f"TomatoYOLODataset.update_labels_info: 从边界框提取cluster_ids，形状={label['cluster_ids'].shape}")
            
            # 处理h_rel (第6列)
            if bbox_shape[1] >= 6 and "h_rel" not in label and self.has_h_rel:
                if isinstance(label["bboxes"], np.ndarray):
                    label["h_rel"] = label["bboxes"][:, 5:6]
                else:  # torch.Tensor
                    label["h_rel"] = label["bboxes"][:, 5:6].clone()  # 使用clone防止共享内存
                LOGGER.debug(f"TomatoYOLODataset.update_labels_info: 从边界框提取h_rel，形状={label['h_rel'].shape}")
            
            # 更新边界框为标准4列格式
            label["bboxes"] = bboxes_standard
            LOGGER.debug(f"TomatoYOLODataset.update_labels_info: 更新边界框为标准格式，新形状={label['bboxes'].shape}")
        
        # 确保cluster_ids和h_rel字段存在（如果需要）
        if self.has_cluster_ids and "cluster_ids" not in label:
            # 如果指定需要cluster_ids但标签中没有，创建默认值
            num_boxes = label["bboxes"].shape[0]
            if isinstance(label["bboxes"], np.ndarray):
                label["cluster_ids"] = np.zeros((num_boxes, 1), dtype=np.float32)
            else:  # torch.Tensor
                label["cluster_ids"] = torch.zeros((num_boxes, 1), dtype=torch.float32)
            LOGGER.debug(f"TomatoYOLODataset.update_labels_info: 创建默认cluster_ids，形状={label['cluster_ids'].shape}")
        
        if self.has_h_rel and "h_rel" not in label:
            # 如果指定需要h_rel但标签中没有，创建默认值
            num_boxes = label["bboxes"].shape[0]
            if isinstance(label["bboxes"], np.ndarray):
                label["h_rel"] = np.zeros((num_boxes, 1), dtype=np.float32)
            else:  # torch.Tensor
                label["h_rel"] = torch.zeros((num_boxes, 1), dtype=torch.float32)
            LOGGER.debug(f"TomatoYOLODataset.update_labels_info: 创建默认h_rel，形状={label['h_rel'].shape}")
        
        # 确保cluster_ids和h_rel是正确的数组类型和形状
        if "cluster_ids" in label:
            if isinstance(label["cluster_ids"], list):
                label["cluster_ids"] = np.array(label["cluster_ids"]).reshape(-1, 1)
            elif isinstance(label["cluster_ids"], np.ndarray) and label["cluster_ids"].ndim == 1:
                label["cluster_ids"] = label["cluster_ids"].reshape(-1, 1)
            elif isinstance(label["cluster_ids"], torch.Tensor) and label["cluster_ids"].ndim == 1:
                label["cluster_ids"] = label["cluster_ids"].reshape(-1, 1)
        
        if "h_rel" in label:
            if isinstance(label["h_rel"], list):
                label["h_rel"] = np.array(label["h_rel"]).reshape(-1, 1)
            elif isinstance(label["h_rel"], np.ndarray) and label["h_rel"].ndim == 1:
                label["h_rel"] = label["h_rel"].reshape(-1, 1)
            elif isinstance(label["h_rel"], torch.Tensor) and label["h_rel"].ndim == 1:
                label["h_rel"] = label["h_rel"].reshape(-1, 1)
        
        # 调用父类方法处理基本标签
        label = super().update_labels_info(label)
        
        # 记录处理后的字段
        LOGGER.debug(f"TomatoYOLODataset.update_labels_info: 处理后标签字段: {list(label.keys())}")
        
        return label
        
    @staticmethod
    def collate_fn(batch):
        """
        将数据样本整合为批次
        
        Args:
            batch: 数据样本列表
            
        Returns:
            dict: 批次数据字典
        """
        if not batch:
            return {}
            
        # 创建新的批次字典
        new_batch = defaultdict(list)
        for sample in batch:
            for key, value in sample.items():
                new_batch[key].append(value)
        
        for k, value in new_batch.items():
            if k == "img":
                # 图像需要堆叠
                value = torch.stack(value, 0)
            elif k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                # 这些字段需要连接
                try:
                    value = torch.cat(value, 0)
                except Exception as e:
                    # 如果连接失败，可能是张量形状不兼容，尝试转换
                    LOGGER.warning(f"TomatoYOLODataset.collate_fn: 连接{k}失败: {e}，尝试修复")
                    # 检查是否所有元素都是张量
                    all_tensors = all(isinstance(v, (torch.Tensor, np.ndarray)) for v in value)
                    if not all_tensors:
                        LOGGER.warning(f"TomatoYOLODataset.collate_fn: {k}包含非张量元素，跳过")
                        continue
                    
                    # 如果都是张量，尝试转换为兼容形状
                    tensors = []
                    for v in value:
                        if isinstance(v, np.ndarray):
                            v = torch.from_numpy(v)
                        # 确保是2D张量
                        if v.ndim == 1:
                            v = v.unsqueeze(-1)
                        tensors.append(v)
                    
                    # 再次尝试连接
                    try:
                        value = torch.cat(tensors, 0)
                    except Exception as e2:
                        LOGGER.error(f"TomatoYOLODataset.collate_fn: 修复后连接{k}仍然失败: {e2}")
                        # 失败则使用空张量
                        value = torch.zeros((0, 1), dtype=torch.float32)
            elif k == "cluster_ids":
                # 番茄串ID需要在批次内保持唯一，逐图像累加偏移
                try:
                    tensors = []
                    offset = 0
                    for v in value:
                        if v is None:
                            v = torch.zeros((0, 1), dtype=torch.float32)
                        elif isinstance(v, np.ndarray):
                            v = torch.from_numpy(v)
                        elif not isinstance(v, torch.Tensor):
                            try:
                                v = torch.tensor(v, dtype=torch.float32)
                            except Exception as e:
                                LOGGER.warning(f"TomatoYOLODataset.collate_fn: 转换{k}失败: {e}，使用空张量")
                                v = torch.zeros((0, 1), dtype=torch.float32)

                        if v.ndim == 1:
                            v = v.reshape(-1, 1)

                        if v.numel():
                            mask = v >= 0
                            if mask.any():
                                v = v.clone()
                                v[mask] += offset
                                offset = int(v[mask].max().item()) + 1
                        tensors.append(v)

                    value = torch.cat(tensors, 0)
                    LOGGER.debug(
                        f"TomatoYOLODataset.collate_fn: 成功连接{k}，形状={value.shape}, 最终偏移={offset}"
                    )
                except Exception as e:
                    LOGGER.error(f"TomatoYOLODataset.collate_fn: 处理{k}时发生错误: {e}")
                    value = torch.zeros((0, 1), dtype=torch.float32)
            elif k == "h_rel":
                # h_rel字段直接连接
                try:
                    tensors = []
                    for v in value:
                        if v is None:
                            v = torch.zeros((0, 1), dtype=torch.float32)
                        elif isinstance(v, np.ndarray):
                            v = torch.from_numpy(v)
                        elif not isinstance(v, torch.Tensor):
                            try:
                                v = torch.tensor(v, dtype=torch.float32)
                            except Exception as e:
                                LOGGER.warning(f"TomatoYOLODataset.collate_fn: 转换{k}失败: {e}，使用空张量")
                                v = torch.zeros((0, 1), dtype=torch.float32)

                        if v.ndim == 1:
                            v = v.reshape(-1, 1)

                        tensors.append(v)

                    value = torch.cat(tensors, 0)
                    LOGGER.debug(f"TomatoYOLODataset.collate_fn: 成功连接{k}，形状={value.shape}")
                except Exception as e:
                    LOGGER.error(f"TomatoYOLODataset.collate_fn: 处理{k}时发生错误: {e}")
                    value = torch.zeros((0, 1), dtype=torch.float32)
            elif k in ('ratio_pad', 'im_file', 'ori_shape', 'resized_shape'):
                pass
            
            # 将处理后的值添加到新批次
            new_batch[k] = value
            
        # 处理批次索引
        if "batch_idx" in new_batch:
            batch_idx_list = list(new_batch["batch_idx"])
            for i in range(len(batch_idx_list)):
                batch_idx_list[i] += i  # 添加目标图像索引
            new_batch["batch_idx"] = torch.cat(batch_idx_list, 0)
        else:
            # 如果没有batch_idx，但有bboxes，创建一个
            if "bboxes" in new_batch and len(new_batch["bboxes"]) > 0:
                # 计算每个样本的边界框数量
                box_counts = []
                for b in batch:
                    if "bboxes" in b and isinstance(b["bboxes"], torch.Tensor):
                        box_counts.append(len(b["bboxes"]))
                    else:
                        box_counts.append(0)
                
                # 创建batch_idx
                batch_idx = []
                for i, count in enumerate(box_counts):
                    batch_idx.append(torch.full((count,), i, dtype=torch.float32))
                
                if batch_idx:  # 确保有内容
                    new_batch["batch_idx"] = torch.cat(batch_idx, 0)
        
        # 打印调试信息
        if "bboxes" in new_batch:
            LOGGER.debug(f"TomatoYOLODataset.collate_fn: bboxes形状={new_batch['bboxes'].shape}")
        if "cluster_ids" in new_batch:
            LOGGER.debug(f"TomatoYOLODataset.collate_fn: cluster_ids形状={new_batch['cluster_ids'].shape}")
        if "h_rel" in new_batch:
            LOGGER.debug(f"TomatoYOLODataset.collate_fn: h_rel形状={new_batch['h_rel'].shape}")
        
        # 打印准备目标数据的信息
        fields = ["im_file", "ori_shape", "resized_shape", "cluster_ids", "h_rel", "bboxes", "batch_idx", "img", "cls"]
        present_fields = [f for f in fields if f in new_batch]
        LOGGER.info(f"准备目标数据，批次字段: {present_fields}")
        
        # 检查特殊字段
        for field in ["batch_idx", "cls", "bboxes"]:
            if field in new_batch:
                tensor = new_batch[field]
                shape_info = tensor.shape if hasattr(tensor, 'shape') else None
                dtype_info = tensor.dtype if hasattr(tensor, 'dtype') else None
                range_info = "[N/A, N/A]"
                if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
                    range_info = f"[{tensor.min().item():.4f}, {tensor.max().item():.4f}]"
                LOGGER.info(f"{field}: 形状={shape_info}, 类型={dtype_info}, 值范围={range_info}")
        
        # 检查批次中是否有有效标签
        if "bboxes" in new_batch and new_batch["bboxes"].numel() == 0:
            LOGGER.warning("批次中没有有效的标签或边界框")
        
        return new_batch


class YOLOMultiModalDataset(YOLODataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        task (str): An explicit arg to point current task, Defaults to 'detect'.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """

    def __init__(self, *args, data=None, task="detect", **kwargs):
        """Initializes the YOLOMultiModalDataset with optional configurations for task specific settings."""
        super().__init__(*args, data=data, task=task, **kwargs)

    def update_labels_info(self, label):
        """Custom your label format here."""
        label = super().update_labels_info(label)
        return label

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        transforms = super().build_transforms(hyp)
        transforms.insert(0, RandomLoadText())
        return transforms


class GroundingDataset(YOLODataset):
    """YOLO Grounding Dataset."""

    def __init__(self, *args, task="detect", json_file, **kwargs):
        """Initializes the GroundingDataset with optional configurations for segments and keypoints."""
        self.json_file = json_file
        super().__init__(*args, task=task, **kwargs)

    def get_img_files(self, img_path):
        """Read image files."""
        json_file = str(self.json_file)
        with open(json_file) as f:
            json_data = json.load(f)["images"]
        return [os.path.join(Path(json_file).parent, x["file_name"]) for x in json_data]

    def get_labels(self):
        """
        Returns dictionary of labels for all images in dataset.

        Returns:
            (dict): Dictionary of labels.
        """
        json_file = str(self.json_file)
        with open(json_file, "r") as f:
            train_json = json.load(f)

        image_dict = {x["id"]: x for x in train_json["images"]}
        category_dict = {x["id"]: x for x in train_json["categories"]}
        # categories = [cat["name"] for cat in train_json["categories"]]
        self.image_ids = [image["id"] for image in train_json["images"]]
        self.annotations = defaultdict(list)
        for ann in train_json["annotations"]:
            self.annotations[ann["image_id"]].append(ann)
        # self.oids = set()
        self.inner_text_dict = defaultdict(list)  # Each element is a list: [text, [], [box_id, label_id]]
        for image_id in self.image_ids:
            info = image_dict[image_id]
            img_path = os.path.join(str(Path(json_file).parent), info["file_name"])

            # Get image dimensions to convert relative box coordinates to absolute coordinates.
            # Note that instances without boxes and with polygonal segmentations will require the
            # image dimensions as well. These are width and height.
            # width, height = info["width"], info["height"]
            for ann in self.annotations[image_id]:
                category_id = int(ann["category_id"])
                assert category_id in category_dict, f"Category {category_id} not found in categories"
                # For visualization purpose, we need the original category name
                # category_name = category_dict[category_id]["name"]
                nbox = ann["bbox"]
                for text in ann["texts"]:
                    self.inner_text_dict[img_path].append([text, []])
                    self.inner_text_dict[img_path][-1][1].append(
                        nbox + [category_id, 1]
                    )  # Grounding json format [xyxy, category_id, score]
                    # self.oids.add(text)

        labels = {}
        # Build a dictionary mapping img_path to [text, [bboxes]], where bboxes are in (xyxy, category_id, score).
        for img_path, text_bboxes in self.inner_text_dict.items():
            texts = []
            boxes = []
            for text, bboxes in text_bboxes:
                texts.append(text)
                boxes.append(torch.tensor(bboxes, dtype=torch.float32))

            labels[img_path] = {"texts": texts, "bboxes": boxes}

        return labels

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        transforms = super().build_transforms(hyp)
        transforms.insert(0, RandomLoadText())
        return transforms


class YOLOConcatDataset(ConcatDataset):
    """
    YOLOConcatDataset class for backward compatibility with older YOLO versions.

    This interface preserves the old behavior of the ConcatDataset in YOLO, ensuring data can be
    properly combined for training models.
    """

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        return YOLODataset.collate_fn(batch)


class SemanticDataset(BaseDataset):
    """
    Dataset for semantic segmentation.

    Args:
        data (dict): A dataset YAML dictionary. Defaults to None.
        task (str): An explicit arg to point current task. Defaults to 'segment'.

    Returns:
        (PyTorch Dataset): A PyTorch dataset object that can be used for training semantic segmentation model.
    """

    def __init__(self):
        """Initialize instance with YAML file containing configuration variables."""
        raise NotImplementedError("SemanticDataset is not yet implemented.")


class ClassificationDataset:
    """
    Dataset for image classification tasks.

    This class handles the loading and processing of image classification data, including functionalities for training
    with various augmentations, caching images for faster training, and organizing data into appropriate formats for
    model training in classification tasks.

    Args:
        root (str): Path to the root directory of the dataset.
        args (dict): Configuration for the dataset. Includes settings like image size and augmentation parameters.
        augment (bool, optional): Whether to use data augmentation. Defaults to False.
        prefix (str, optional): Prefix added to logging messages for debug info. Defaults to "".

    Example:
        ```python
        dataset = ClassificationDataset(root='path/to/data', args={'imgsz': 640})
        for item in dataset:
            images, labels = item
            # Process images and labels for training
        ```

    Attributes:
        root (Path): The root directory path containing the dataset.
        im_files (list): List of paths to the image files.
        nf (int): Number of image files.
        args (dict): Arguments for dataset configuration.
        [and more...]
    """

    def __init__(self, root, args, augment=False, prefix=""):
        """Initialize the ClassificationDataset class with root directory path and image size for processing."""
        self.root = Path(root)
        self.im_files = sorted(
            [
                f for f in (self.root.parent if self.root.is_file() else self.root).rglob("*.*")
                if f.suffix[1:].lower() in IMG_FORMATS
            ],
            key=lambda x: str(x),
        )
        self.nf = len(self.im_files)  # number of image files
        if not self.nf:
            msg = "No images found in directory" if self.root.is_dir() else "Invalid path"
            raise FileNotFoundError(f"{prefix}{msg}: {self.root}")
        self.args = args
        self.prefix = prefix
        self.augment = augment
        self.torch_transforms = classify_transforms(
            self.args.imgsz, image_weights=self.args.image_weights, auto=self.args.auto_augment
        )
        self.cache = self.args.cache_images == "ram"
        self.imgs = [None] * self.nf
        self.classes = []
        for file in self.im_files:
            parent = str(file.parent).split(os.sep)
            if len(parent):
                self.classes.append(parent[-1])
        self.classes = sorted(list(set(self.classes)))  # classes list
        self.labels = np.array([self.classes.index(c) for c in self.classes])  # classes indices
        if self.cache:
            self.imgs, self.im_hw0, self.im_hw = [None] * self.nf, [None] * self.nf, [None] * self.nf
            self.npy_files = [Path(f.with_suffix(".npy")) for f in self.im_files]
            self.cache_images()

    def __getitem__(self, i):
        """Returns (image, label) for i-th item."""
        if self.cache:
            f = self.im_files[i]
            parent = str(f.parent).split(os.sep)
            idx = self.classes.index(parent[-1])
            img = self.imgs[i]

        else:
            f = self.im_files[i]
            parent = str(f.parent).split(os.sep)
            idx = self.classes.index(parent[-1])
            img = cv2.imread(str(f))[..., ::-1]
        return self.torch_transforms(image=img)["image"], idx

    def __len__(self) -> int:
        """Return number of images."""
        return len(self.im_files)

    def verify_images(self):
        """Verify all images in dataset."""
        desc = f"{self.prefix}Scanning {self.root}..."
        path = f"{self.root}"
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(func=verify_image, iterable=zip(self.im_files, repeat(True), repeat(path)))
            pbar = TQDM(results, desc=desc, total=self.nf)
            for file, status in pbar:
                if not status:
                    LOGGER.warning(f"{self.prefix}Warning ⚠️ Ignoring corrupted image: {file}") 

    def cache_labels(self, path=Path("./labels.cache")):
        """
        缓存数据集标签，检查图像并读取形状。
        
        Args:
            path (Path): 保存缓存文件的路径。默认为Path("./labels.cache")。
            
        Returns:
            (dict): 标签。
        """
        # 首先调用YOLODataset的cache_labels方法
        try:
            # 尝试使用原始方法
            return super().cache_labels(path)
        except Exception as e:
            LOGGER.warning(f"使用原始cache_labels方法失败: {e}，尝试自定义方法")
            
        # 如果失败，使用自定义方法
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        
        # 确保self.data存在
        if self.data is None:
            self.data = {"names": ["tomato"], "nc": 1}
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ self.data is None, using default configuration.")
        
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(nkpt),
                    repeat(ndim),
                    repeat(False),  # use_keypoints
                    repeat(False),  # use_segments
                    repeat(False),  # use_obb
                    repeat(self.has_cluster_ids),  # has_cluster_id
                    repeat(self.has_h_rel),  # has_h_rel
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],  # n, 1
                            "bboxes": lb[:, 1:5],  # n, 4
                            "segments": segments,
                            "keypoints": keypoint,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                    # 添加cluster_ids和h_rel字段
                    if self.has_cluster_ids and lb.shape[1] > 5:
                        x["labels"][-1]["cluster_ids"] = lb[:, 5:6]
                    if self.has_h_rel and lb.shape[1] > 6:
                        x["labels"][-1]["h_rel"] = lb[:, 6:7]
                        
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        
        # 添加额外的日志，确保用户了解实际情况
        LOGGER.info(f"缓存统计: 总图像数={len(self.im_files)}, 有标签图像数={nf}, 无标签图像数={nm + ne}, 损坏图像数={nc}")
        
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x 