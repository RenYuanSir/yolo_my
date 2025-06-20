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
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data or {}
        
        # Support for tomato fruit cluster and relative position
        self.has_cluster_id = self.data.get("has_cluster_id", False)
        self.has_h_rel = self.data.get("has_h_rel", False)
        
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        super().__init__(*args, **kwargs)

    def cache_labels(self, path=Path("./labels.cache")):
        """Cache labels (segments, polygons, etc) to RAM for faster training."""
        x = {"labels": []}
        nm, nf, ne, nc, ms = 0, 0, 0, 0, 0  # number missing, found, empty, corrupt, multiscale
        desc = f"{self.prefix}Scanning {path.parent / path.name}..."
        total = len(self.im_files)
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    [self.prefix] * total,
                    [(self.data or {}).get("kpt_shape", (0, 0))[0]] * total,
                    [len(self.label_files[0]) > 0] * total,
                    [self.use_keypoints] * total,
                    [self.use_segments] * total,
                    [self.use_obb] * total,
                    [(self.data or {}).get("has_cluster_id", False)] * total,
                    [(self.data or {}).get("has_h_rel", False)] * total,
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, ne_f, nc_f, ms_f in pbar:
                if nm_f is None:  # Handle case where nm_f might be None
                    nm_f = 0
                if ne_f is None:
                    ne_f = 0
                if nc_f is None:
                    nc_f = 0
                if ms_f is None:
                    ms_f = 0
                    
                nm += nm_f
                ne += ne_f
                nc += nc_f
                ms += ms_f
                if im_file:
                    # Handle 7-column format for tomato dataset (class, x, y, w, h, cluster_id, h_rel)
                    # or 6-column format (class, x, y, w, h, cluster_id)
                    label_dict = {
                        "im_file": im_file,
                        "shape": shape,
                        "cls": lb[:, 0:1],
                        "normalized": True,
                        "bbox_format": "xywh",
                    }
                    
                    # Standard YOLO format is 5 columns
                    if lb.shape[1] >= 5:
                        label_dict["bboxes"] = lb[:, 1:5]
                        
                    # Handle extra columns if present
                    if lb.shape[1] >= 6:  # Has cluster_id
                        label_dict["cluster_ids"] = lb[:, 5:6]
                        
                    if lb.shape[1] >= 7:  # Has h_rel
                        label_dict["h_rel"] = lb[:, 6:7]
                        
                    # Add segments and keypoints if present
                    label_dict["segments"] = segments
                    label_dict["keypoints"] = keypoint
                    
                    x["labels"].append(label_dict)
                else:
                    nf += 1
                pbar.desc = f"{desc} {nf} images found, {nm} missing, {ne} empty, {nc} corrupt, {ms} multiscale"
            pbar.close()

        if nf == 0:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        try:
            x["results"] = nf, nm, ne, nc, len(self.im_files)
            torch.save(x, path)
            LOGGER.info(f"New cache created: {path}")
        except Exception as e:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable: {e}")
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
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            LOGGER.info(f"{d}, {TQDM(position=1, disable=True)}")
        assert (
            nf > 0
        ), f"No labels found in {cache_path}, can not start training. {HELP_URL}"

        # Read cache
        [cache.pop(k) for k in ("hash", "version")]  # remove items
        labels = cache  # dictionary of labels, i.e. labels = {im_file: [cls, x, y, w, h, ...]}
        if self.default_transforms:
            for k in labels:
                if len(labels[k]):
                    if (
                        labels[k][:, 0] > 0
                    ).all():  # check if all classes > 0, i.e. no background class in YOLO format labels
                        labels[k][:, 0] -= 1  # background/non-background classes starts from 0 in AUTO transforms

        if self.fraction < 1:
            for k in list(labels.keys())[: round(len(labels) * (1.0 - self.fraction))]:
                if random.random() < 1 - self.fraction/(1 - self.fraction + 1E-10):
                    labels.pop(k)
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
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
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