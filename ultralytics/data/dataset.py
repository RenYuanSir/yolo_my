# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json
from collections import defaultdict
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset

from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM, colorstr
from ultralytics.utils.ops import resample_segments
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
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    save_dataset_cache_file,
    verify_image,
    verify_image_label,
)

# Ultralytics dataset *.cache version, >= 1.0.0 for YOLOv8
DATASET_CACHE_VERSION = "1.0.3"


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
        # ① 明确各功能开关 —— 不传给父类，只留作属性
        self.use_segments  = task == "segment"
        self.use_keypoints = False            # 彻底关闭
        self.use_obb       = task == "obb"

        # ② 数据集自定义标志
        self.data = data.copy() if isinstance(data, dict) else {}
        self.has_cluster_id = self.data.get("has_cluster_id", False)
        self.has_h_rel      = self.data.get("has_h_rel", False)
        LOGGER.info(f"初始化数据集: has_cluster_id={self.has_cluster_id}, has_h_rel={self.has_h_rel}")

        # ③ 调用父类 —— 只把 data 放进去，其余多余键全部丢弃
        super().__init__(*args, data=self.data, **kwargs)

    def cache_labels(self, path: Path = Path("./labels.cache")):
        """
        Parse all label files once, verify + preprocess，缓存到磁盘：
        x = {
            labels: [dict ...]         # 每张图片一个字典
            hash: <sha256>,
            version: <int>,
            results: (nf, nm, ne, nc, total)
        }
        """
        x = {"labels": []}
        nm = nf = ne = nc = ms = 0        # missing / found / empty / corrupt / multiscale
        total = len(self.im_files)
        desc = f"{self.prefix}Scanning {path.parent / path.name}..."
        
        # ── 读取 YAML 中的扩展标志 ────────────────────────────────────────────────
        data_dict = self.data or {}
        has_cluster_id = bool(data_dict.get("has_cluster_id", False))
        has_h_rel      = bool(data_dict.get("has_h_rel", False))
        kpt_shape      = data_dict.get("kpt_shape", (0, 0))
        LOGGER.info(f"Dataset flags →  has_cluster_id={has_cluster_id}, has_h_rel={has_h_rel}")
        
        try:
            with ThreadPool(NUM_THREADS) as pool:
                results = pool.map(
                    verify_image_label,
                    zip(
                        self.im_files,
                        self.label_files,
                        [self.prefix] * total,
                        [kpt_shape[0]] * total,
                        [len(self.label_files) > 0] * total,
                        [self.use_keypoints] * total,
                        [self.use_segments] * total,
                        [self.use_obb] * total,
                        [has_cluster_id] * total,
                        [has_h_rel] * total,
                    ),
                )
    
                pbar = TQDM(results, total=total, desc=desc, unit="image")
                for im_file, lb, shape, segs, kpts, nm_f, ne_f, nc_f, ms_f in pbar:
                    label_dict = {
                        "im_file": im_file,
                        "normalized": True,
                        "bbox_format": "xywh",
                    }
    
                    if im_file and lb is not None and lb.size:
                        label_dict["cls"]    = lb[:, :1]
                        label_dict["bboxes"] = lb[:, 1:5]
    
                        # 确保 cluster_id 和 h_rel 字段
                        n = lb.shape[0]
                        if self.has_cluster_id and lb.shape[1] >= 6:
                            label_dict["cluster_ids"] = lb[:, 5:6]
                        else:
                            # 如果 cluster_id 缺失，填充为零，并确保维度与 cls 和 bboxes 一致
                            label_dict["cluster_ids"] = np.zeros((n, 1), dtype=np.float32)
                        
                        if self.has_h_rel and lb.shape[1] >= 7:
                            label_dict["h_rel"] = lb[:, 6:7]
                        else:
                            # 如果 h_rel 缺失，填充为零，并确保维度与 cls 和 bboxes 一致
                            label_dict["h_rel"] = np.zeros((n, 1), dtype=np.float32)
    
                        nf += 1
                    else:
                        # 如果标签为空，使用默认值
                        label_dict["cls"]    = np.zeros((0, 1), dtype=np.float32)
                        label_dict["bboxes"] = np.zeros((0, 4), dtype=np.float32)
                        if self.has_cluster_id:
                            label_dict["cluster_ids"] = np.zeros((0, 1), dtype=np.float32)
                        if self.has_h_rel:
                            label_dict["h_rel"] = np.zeros((0, 1), dtype=np.float32)
    
                    # 确保所有标签字段存在
                    label_dict.setdefault("cluster_ids", np.zeros((0, 1), dtype=np.float32))
                    label_dict.setdefault("h_rel", np.zeros((0, 1), dtype=np.float32))
    
                    x["labels"].append(label_dict)
                    pbar.set_description(f"{desc} {nf} found, {nm} missing, {ne} empty, {nc} corrupt, {ms} multiscale")
                pbar.close()
    
        except Exception as e:
            LOGGER.error(f"{self.prefix}ERROR during label caching: {e}")
            for im_file in self.im_files:
                x["labels"].append({
                    "im_file": im_file,
                    "shape": (640, 640),
                    "cls": np.zeros((0, 1), dtype=np.float32),
                    "bboxes": np.zeros((0, 4), dtype=np.float32),
                    "normalized": True,
                    "bbox_format": "xywh",
                    "segments": [],
                    "keypoints": None,
                    **({"cluster_ids": np.zeros((0, 1))} if has_cluster_id else {}),
                    **({"h_rel": np.zeros((0, 1))} if has_h_rel else {}),
                })
        
        # ── 完成统计 & 写缓存 ───────────────────────────────────────────────
        if nf == 0:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ No valid labels found in {path}. {HELP_URL}")
        
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["version"] = DATASET_CACHE_VERSION
        x["results"] = (nf, nm, ne, nc, len(self.im_files))
        
        try:
            torch.save(x, path)
            LOGGER.info(f"New cache created: {path}")
        except Exception as e:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ Could not write cache to {path}: {e}")
        
        return x




    def get_labels(self):
        """
        Returns dictionary of labels for all images in dataset.

        Returns:
            (dict): Dictionary of labels.
        """
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            # 尝试多种方式加载缓存文件
            try:
                cache = torch.load(cache_path)
                exists = True
            except:
                try:
                    cache = np.load(cache_path, allow_pickle=True)
                    # 处理numpy数组的情况
                    if hasattr(cache, "item") and callable(cache.item):
                        cache = cache.item()
                    exists = True
                except:
                    raise FileNotFoundError("无法加载缓存文件")
            
            # 检查缓存是否有必要的键，版本是否匹配
            if "version" not in cache or cache["version"] != DATASET_CACHE_VERSION:
                raise AssertionError("Cache version mismatch")
            
            if "hash" not in cache or cache["hash"] != get_hash(self.label_files + self.im_files):
                raise AssertionError("Cache hash mismatch")
        
        except (FileNotFoundError, AssertionError, AttributeError, KeyError) as e:
            LOGGER.info(f"重建缓存: {e}")
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            LOGGER.info(f"{d}, {TQDM(position=1, disable=True)}")
        assert (
            nf > 0
        ), f"No labels found in {cache_path}, can not start training. {HELP_URL}"

        # 安全地删除可能存在的键
        for k in ("hash", "version"):
            if k in cache:
                cache.pop(k)

        # 确保我们返回的是在cache_labels函数中创建的"labels"列表，而不是整个缓存对象
        if "labels" in cache and isinstance(cache["labels"], list):
            labels = cache["labels"]
        else:
            # 如果缓存格式不符合期望，我们需要重建缓存
            LOGGER.warning("缓存格式不是预期的格式，重新构建缓存...")
            cache, _ = self.cache_labels(cache_path)
            labels = cache["labels"]

        if self.fraction < 1:
            n_labels = len(labels)
            indices = list(range(n_labels))
            random.shuffle(indices)
            keep_count = max(1, int(n_labels * self.fraction))
            indices = indices[:keep_count]
            labels = [labels[i] for i in indices]

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

    def __getitem__(self, index):
        """
        Returns a single sample with augmentations.
        This simplified version correctly packages data for the transform pipeline.
        """
        # This will now correctly pass a dictionary with Instances object to transforms
        label = self.get_image_and_label(index)
        
        # Apply transforms
        return self.transforms(label) if self.augment else label


    def update_labels_info(self, label):
        """
        Packages label data into a dictionary with an Instances object.
        This is the crucial step to ensure data survives augmentations.
        """
        bboxes = label.pop("bboxes")
        cls = label.pop("cls")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)

        # --- Safely pop custom attributes ---
        cluster_ids = label.pop("cluster_ids", None)
        h_rel = label.pop("h_rel", None)

        # Create the Instances object, passing custom attributes to its constructor
        instances = Instances(bboxes, segments, keypoints, bbox_format="xywh", normalized=True,
                              cluster_ids=cluster_ids, h_rel=h_rel)
        
        label["instances"] = instances
        label["cls"] = cls
        return label


    @staticmethod
    def collate_fn(batch):
        """
        A standard and simple collate function.
        It expects the 'Format' transform to have created a 'labels' tensor.
        """
        new_batch = {}
        # Get keys from the first sample
        keys = batch[0].keys()
        for key in keys:
            if key == 'labels':
                # For labels, concatenate them along the first dimension
                new_batch[key] = torch.cat([b[key] for b in batch if b[key] is not None and len(b[key]) > 0], 0)
            elif key == 'img':
                # For images, stack them to create a batch dimension
                new_batch[key] = torch.stack([b['img'] for b in batch], 0)
            else:
                # For other metadata like file paths, just collect them in a list
                new_batch[key] = [b[key] for b in batch]

        # Ensure 'labels' key exists even if the batch has no labels
        if 'labels' not in new_batch:
            new_batch['labels'] = torch.zeros((0, 8)) # [batch_idx, cls, xywh, cluster_id, h_rel]
            
        # Add batch_idx to the labels tensor
        if len(new_batch['labels']) > 0:
            batch_idx = torch.cat([torch.full((b['labels'].shape[0], 1), i) for i, b in enumerate(batch) if b['labels'] is not None and len(b['labels']) > 0])
            new_batch['labels'] = torch.cat((batch_idx, new_batch['labels']), 1)

        return new_batch


class TomatoYOLODataset(YOLODataset):
    """
    The custom dataset class is now much simpler. It just needs to
    set the flags and can inherit most of the logic.
    """
    def __init__(self, *args, data=None, **kwargs):
        data = data or {}
        data["has_cluster_id"] = True
        data["has_h_rel"] = True
        super().__init__(*args, data=data, **kwargs)
        LOGGER.info(f"TomatoYOLODataset initialized with: has_cluster_id={self.has_cluster_id}, has_h_rel={self.has_h_rel}")


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