# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from collections import abc
from itertools import repeat
from numbers import Number
from typing import List

import numpy as np
import torch

from ultralytics.utils import LOGGER
from .ops import ltwh2xywh, ltwh2xyxy, resample_segments, xywh2ltwh, xywh2xyxy, xyxy2ltwh, xyxy2xywh


def _ntuple(n):
    """From PyTorch internals."""

    def parse(x):
        """Parse bounding boxes format between XYWH and LTWH."""
        return x if isinstance(x, abc.Iterable) else tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)
to_4tuple = _ntuple(4)

# `xyxy` means left top and right bottom
# `xywh` means center x, center y and width, height(YOLO format)
# `ltwh` means left top and width, height(COCO format)
_formats = ["xyxy", "xywh", "ltwh"]

__all__ = ("Bboxes", "Instances")  # tuple or list


class Bboxes:
    """
    A class for handling bounding boxes.

    The class supports various bounding box formats like 'xyxy', 'xywh', and 'ltwh'.
    Bounding box data should be provided in numpy arrays.

    Attributes:
        bboxes (numpy.ndarray): The bounding boxes stored in a 2D numpy array.
        format (str): The format of the bounding boxes ('xyxy', 'xywh', or 'ltwh').

    Note:
        This class does not handle normalization or denormalization of bounding boxes.
    """

    def __init__(self, bboxes, format="xyxy") -> None:
        """Initializes the Bboxes class with bounding box data in a specified format."""
        assert format in _formats, f"Invalid bounding box format: {format}, format must be one of {_formats}"
        bboxes = bboxes[None, :] if bboxes.ndim == 1 else bboxes
        assert bboxes.ndim == 2
        assert bboxes.shape[1] == 4
        self.bboxes = bboxes
        self.format = format
        # self.normalized = normalized

    def convert(self, format):
        """Converts bounding box format from one type to another."""
        assert format in _formats, f"Invalid bounding box format: {format}, format must be one of {_formats}"
        if self.format == format:
            return
        elif self.format == "xyxy":
            func = xyxy2xywh if format == "xywh" else xyxy2ltwh
        elif self.format == "xywh":
            func = xywh2xyxy if format == "xyxy" else xywh2ltwh
        else:
            func = ltwh2xyxy if format == "xyxy" else ltwh2xywh
        self.bboxes = func(self.bboxes)
        self.format = format

    def areas(self):
        """Return box areas."""
        return (
            (self.bboxes[:, 2] - self.bboxes[:, 0]) * (self.bboxes[:, 3] - self.bboxes[:, 1])  # format xyxy
            if self.format == "xyxy"
            else self.bboxes[:, 3] * self.bboxes[:, 2]  # format xywh or ltwh
        )

    # def denormalize(self, w, h):
    #    if not self.normalized:
    #         return
    #     assert (self.bboxes <= 1.0).all()
    #     self.bboxes[:, 0::2] *= w
    #     self.bboxes[:, 1::2] *= h
    #     self.normalized = False
    #
    # def normalize(self, w, h):
    #     if self.normalized:
    #         return
    #     assert (self.bboxes > 1.0).any()
    #     self.bboxes[:, 0::2] /= w
    #     self.bboxes[:, 1::2] /= h
    #     self.normalized = True

    def mul(self, scale):
        """
        Multiply bounding box coordinates by scale factor(s).

        Args:
            scale (int | tuple | list): Scale factor(s) for four coordinates.
                If int, the same scale is applied to all coordinates.
        """
        if isinstance(scale, Number):
            scale = to_4tuple(scale)
        assert isinstance(scale, (tuple, list))
        assert len(scale) == 4
        self.bboxes[:, 0] *= scale[0]
        self.bboxes[:, 1] *= scale[1]
        self.bboxes[:, 2] *= scale[2]
        self.bboxes[:, 3] *= scale[3]

    def add(self, offset):
        """
        Add offset to bounding box coordinates.

        Args:
            offset (int | tuple | list): Offset(s) for four coordinates.
                If int, the same offset is applied to all coordinates.
        """
        if isinstance(offset, Number):
            offset = to_4tuple(offset)
        assert isinstance(offset, (tuple, list))
        assert len(offset) == 4
        self.bboxes[:, 0] += offset[0]
        self.bboxes[:, 1] += offset[1]
        self.bboxes[:, 2] += offset[2]
        self.bboxes[:, 3] += offset[3]

    def __len__(self):
        """Return the number of boxes."""
        return len(self.bboxes)

    @classmethod
    def concatenate(cls, boxes_list: List["Bboxes"], axis=0) -> "Bboxes":
        """
        Concatenate a list of Bboxes objects into a single Bboxes object.

        Args:
            boxes_list (List[Bboxes]): A list of Bboxes objects to concatenate.
            axis (int, optional): The axis along which to concatenate the bounding boxes.
                                   Defaults to 0.

        Returns:
            Bboxes: A new Bboxes object containing the concatenated bounding boxes.

        Note:
            The input should be a list or tuple of Bboxes objects.
        """
        assert isinstance(boxes_list, (list, tuple))
        if not boxes_list:
            return cls(np.empty(0))
        assert all(isinstance(box, Bboxes) for box in boxes_list)

        if len(boxes_list) == 1:
            return boxes_list[0]
        return cls(np.concatenate([b.bboxes for b in boxes_list], axis=axis))

    def __getitem__(self, index) -> "Bboxes":
        """
        Retrieve a specific bounding box or a set of bounding boxes using indexing.

        Args:
            index (int, slice, or np.ndarray): The index, slice, or boolean array to select
                                               the desired bounding boxes.

        Returns:
            Bboxes: A new Bboxes object containing the selected bounding boxes.

        Raises:
            AssertionError: If the indexed bounding boxes do not form a 2-dimensional matrix.

        Note:
            When using boolean indexing, make sure to provide a boolean array with the same
            length as the number of bounding boxes.
        """
        if isinstance(index, int):
            return Bboxes(self.bboxes[index].reshape(1, -1))
        b = self.bboxes[index]
        assert b.ndim == 2, f"Indexing on Bboxes with {index} failed to return a matrix!"
        return Bboxes(b)


class Instances:
    """
    Container for bounding boxes, segments, and keypoints of detected objects in an image.

    Attributes:
        _bboxes (Bboxes): Internal object for handling bounding box operations.
        keypoints (ndarray): keypoints(x, y, visible) with shape [N, 17, 3]. Default is None.
        normalized (bool): Flag indicating whether the bounding box coordinates are normalized.
        segments (ndarray): Segments array with shape [N, 1000, 2] after resampling.
        cluster_ids (ndarray): Cluster IDs with shape [N, 1]. Default is None.
        h_rel (ndarray): Relative heights with shape [N, 1]. Default is None.

    Args:
        bboxes (ndarray): An array of bounding boxes with shape [N, 4].
        segments (list | ndarray, optional): A list or array of object segments. Default is None.
        keypoints (ndarray | None): An ndarray with shape [N, 17, 3] or None.
        normalized (bool, optional): Whether the bounding box coordinates are normalized. Default is True.
        bbox_format (str, optional): The format of bounding boxes. Options are 'xyxy', 'xywh', 'ltrb', 'cxcywh'.
                                    Default is 'xywh'.
        cluster_ids (ndarray | None): An array of cluster IDs with shape [N, 1] or None.
        h_rel (ndarray | None): An array of relative heights with shape [N, 1] or None.
    """

    def __init__(
        self,
        bboxes=None,
        segments=None,
        keypoints=None,
        normalized=True, 
        bbox_format="xywh",
        cluster_ids=None,
        h_rel=None,
    ):
        """Initialize the Instances object with bounding boxes, segments, and keypoints."""
        self.normalized = normalized
        self._bboxes = Bboxes(bboxes=bboxes, format=bbox_format)

        # 处理番茄特有属性
        self.cluster_ids = np.array(cluster_ids) if cluster_ids is not None and len(cluster_ids) > 0 else None
        self.h_rel = np.array(h_rel) if h_rel is not None and len(h_rel) > 0 else None

        if segments is None:
            segments = []
        # segments (list or ndarray) to segments_tensor (N, 1000, 2)
        # segments中的每个元素可以是不同长度的ndarray
        self.segments = segments
        if isinstance(segments, list) and len(segments) > 0:
            self.segments = np.stack(segments)
        # segments: [N, 1000, 2]
        elif isinstance(segments, np.ndarray) and segments.ndim == 3:
            self.segments = segments
        else:
            self.segments = np.zeros((0, 1000, 2), dtype=np.float32)

        if keypoints is not None and len(keypoints):
            # for reid use(all keypoint serve as one box)
            self.keypoints = np.array(keypoints, dtype=np.float32)
        else:
            self.keypoints = None

    @property
    def bbox_areas(self):
        """Calculate and return the area of each bounding box."""
        return self._bboxes.areas

    @property
    def bboxes(self):
        """Return the current bounding boxes."""
        return self._bboxes.bboxes

    def scale(self, scale_w, scale_h=None, cur_dim=None):
        """
        Scale the bounding boxes, segments, and keypoints by the provided scaling factors.

        Args:
            scale_w (float): Width scaling factor. If scale_h is None, this value will be used for both width and height.
            scale_h (float | None, optional): Height scaling factor. If None, scale_w will be used for both dimensions.
            cur_dim (tuple | None, optional): Current width and height dimensions.
        
        Returns:
            (Instances): Scaled Instances object.
        """
        if scale_h is None:
            scale_h = scale_w

        self._bboxes.mul(scale=(scale_w, scale_h, scale_w, scale_h))

        if len(self.segments):
            self.segments[..., 0] *= scale_w
            self.segments[..., 1] *= scale_h

        if self.keypoints is not None and len(self.keypoints):
            self.keypoints[..., 0] *= scale_w
            self.keypoints[..., 1] *= scale_h
        # 相对高度h_rel不受图像尺寸影响，不需要缩放
        return self

    def denormalize(self, w, h):
        """
        Denormalize bounding boxes, segments, and keypoints from normalized coordinates.

        Args:
            w (int): Image width.
            h (int): Image height.
        
        Returns:
            (Instances): Denormalized Instances object.
        """
        if not self.normalized:
            return self

        self._bboxes.mul(scale=(w, h, w, h))

        if len(self.segments):
            self.segments[..., 0] *= w
            self.segments[..., 1] *= h

        if self.keypoints is not None and len(self.keypoints):
            self.keypoints[..., 0] *= w
            self.keypoints[..., 1] *= h

        self.normalized = False
        # 相对高度h_rel不受图像尺寸影响，不需要缩放
        return self

    def normalize(self, w, h):
        """
        Normalize bounding boxes, segments, and keypoints to [0, 1] based on image dimensions.

        Args:
            w (int): Image width.
            h (int): Image height.
        
        Returns:
            (Instances): Normalized Instances object.
        """
        if self.normalized:
            return self

        self._bboxes.mul(scale=(1 / w, 1 / h, 1 / w, 1 / h))

        if len(self.segments):
            self.segments[..., 0] /= w
            self.segments[..., 1] /= h

        if self.keypoints is not None and len(self.keypoints):
            self.keypoints[..., 0] /= w
            self.keypoints[..., 1] /= h

        self.normalized = True
        # 相对高度h_rel不受图像尺寸影响，不需要缩放
        return self

    def convert_bbox(self, format):
        """
        Convert the format of the bounding boxes.

        Args:
            format (str): The target format for bounding boxes.
        
        Returns:
            (Instances): Instances object with the bounding boxes in the target format.
        """
        self._bboxes.convert(format)
        return self

    def add_padding(self, padw, padh):
        """
        Add padding to the bounding boxes, segments, and keypoints.

        Args:
            padw (int): Width padding.
            padh (int): Height padding.
        
        Returns:
            (Instances): Instances object with added padding.
        """
        self._bboxes.add(offset=(padw, padh, padw, padh))

        if len(self.segments):
            self.segments[..., 0] += padw
            self.segments[..., 1] += padh

        if self.keypoints is not None and len(self.keypoints):
            self.keypoints[..., 0] += padw
            self.keypoints[..., 1] += padh
        # 相对高度h_rel不受图像尺寸影响，不需要缩放
        return self

    def clip(self, w, h):
        """
        Clip the bounding boxes, segments, and keypoints to image boundaries.

        Args:
            w (int): Image width.
            h (int): Image height.
        
        Returns:
            (Instances): Instances object with clipped coordinates.
        """
        self._bboxes.convert(format="xyxy")
        self._bboxes.bboxes[:, [0, 2]] = self._bboxes.bboxes[:, [0, 2]].clip(0, w)
        self._bboxes.bboxes[:, [1, 3]] = self._bboxes.bboxes[:, [1, 3]].clip(0, h)
        if isinstance(self.segments, np.ndarray) and self.segments.size > 0:
            self.segments[:, :, 0] = self.segments[:, :, 0].clip(0, w)
            self.segments[:, :, 1] = self.segments[:, :, 1].clip(0, h)
        elif self.segments:  # list of np.ndarray
            for i in range(len(self.segments)):
                if self.segments[i].size > 0:
                    self.segments[i][:, 0] = self.segments[i][:, 0].clip(0, w)
                    self.segments[i][:, 1] = self.segments[i][:, 1].clip(0, h)
        if self.keypoints is not None and self.keypoints.size > 0:
            self.keypoints[:, :, 0] = self.keypoints[:, :, 0].clip(0, w)
            self.keypoints[:, :, 1] = self.keypoints[:, :, 1].clip(0, h)
            
        # cluster_ids和h_rel在裁剪后不变
        return self

    def flip(self, fliplr=True, h=0, w=0):
        """
        Flip the bounding boxes, segments, and keypoints horizontally or vertically.

        Args:
            fliplr (bool, optional): Whether to flip horizontally. Default is True.
            h (int, optional): Image height. Default is 0.
            w (int, optional): Image width. Default is 0.
        
        Returns:
            (Instances): Instances object with flipped coordinates.
        """
        self._bboxes.bboxes[:, [0, 2]] = self._bboxes.bboxes[:, [2, 0]] if fliplr else self._bboxes.bboxes[:, [0, 2]]
        self._bboxes.bboxes[:, [1, 3]] = self._bboxes.bboxes[:, [3, 1]] if fliplr else self._bboxes.bboxes[:, [1, 3]]
        if isinstance(self.segments, np.ndarray) and self.segments.size > 0:
            self.segments[:, :, 0] = w - self.segments[:, :, 0] if fliplr else self.segments[:, :, 0]
            self.segments[:, :, 1] = h - self.segments[:, :, 1] if fliplr else self.segments[:, :, 1]
        elif self.segments:  # list of np.ndarray
            for i in range(len(self.segments)):
                if self.segments[i].size > 0:
                    self.segments[i][:, 0] = w - self.segments[i][:, 0] if fliplr else self.segments[i][:, 0]
                    self.segments[i][:, 1] = h - self.segments[i][:, 1] if fliplr else self.segments[i][:, 1]
        if self.keypoints is not None and self.keypoints.size > 0:
            self.keypoints[:, :, 0] = w - self.keypoints[:, :, 0] if fliplr else self.keypoints[:, :, 0]
            self.keypoints[:, :, 1] = h - self.keypoints[:, :, 1] if fliplr else self.keypoints[:, :, 1]
            
        # 处理h_rel - 因为上下翻转会改变高度顺序，所以需要反转相对高度
        if self.h_rel is not None and self.h_rel.size > 0:
            # 保持0.0（串本身）和-1.0（未知）不变，其他值翻转(1.0 - h_rel)
            valid_mask = (self.h_rel != -1.0) & (self.h_rel != 0.0)
            self.h_rel[valid_mask] = 1.0 - self.h_rel[valid_mask]

        return self

    def update(self, bboxes=None, segments=None, keypoints=None, cluster_ids=None, h_rel=None):
        """
        Update the attributes of the Instances object.

        Args:
            bboxes (ndarray | None, optional): New bounding boxes. Default is None.
            segments (list | ndarray | None, optional): New segments. Default is None.
            keypoints (ndarray | None, optional): New keypoints. Default is None.
            cluster_ids (ndarray | None, optional): New cluster IDs. Default is None.
            h_rel (ndarray | None, optional): New relative heights. Default is None.
        
        Returns:
            (Instances): Updated Instances object.
        """
        if bboxes is not None:
            self._bboxes.bboxes = bboxes
            
        if segments is not None:
            if isinstance(segments, list) and len(segments) > 0:
                self.segments = np.stack(segments)
            elif isinstance(segments, np.ndarray):
                self.segments = segments
            else:
                self.segments = np.zeros((0, 1000, 2), dtype=np.float32)

        if keypoints is not None:
            if len(keypoints):
                self.keypoints = np.array(keypoints, dtype=np.float32)
            else:
                self.keypoints = None
                
        # 番茄特有属性更新
        if cluster_ids is not None:
            self.cluster_ids = cluster_ids
            
        if h_rel is not None:
            self.h_rel = h_rel
            
        return self

    def __len__(self):
        """Return the number of instances."""
        return len(self._bboxes)

    @classmethod
    def concatenate(cls, instances_list: List["Instances"], axis=0) -> "Instances":
        """
        Concatenate a list of Instances into one Instances.

        Args:
            instances_list (List[Instances]): A list of Instances objects.
            axis (int): The axis along which to concatenate.

        Returns:
            Instances: A new Instances containing all instances from the list.

        Raises:
            AssertionError: If the list is empty or doesn't entirely consist of Instances objects.

        Examples:
            >>> instances1 = Instances(np.array([[0, 0, 10, 10]]), np.array([[[0, 0], [0, 10], [10, 10], [10, 0]]]))
            >>> instances2 = Instances(np.array([[20, 20, 30, 30]]), np.array([[[20, 20], [20, 30], [30, 30], [30, 20]]]))
            >>> instances = Instances.concatenate([instances1, instances2])
            >>> instances.bboxes
            array([[ 0,  0, 10, 10],
                   [20, 20, 30, 30]])
            >>> instances.segments.shape
            (2, 4, 2)
        """
        assert isinstance(instances_list, (list, tuple))
        if len(instances_list) == 0:
            return cls(np.empty(0))
        assert all(isinstance(item, Instances) for item in instances_list)

        if len(instances_list) == 1:
            return instances_list[0]

        use_keypoint = instances_list[0].keypoints is not None
        use_segments = instances_list[0].segments is not None
        
        # 番茄特有属性
        use_cluster_ids = instances_list[0].cluster_ids is not None
        use_h_rel = instances_list[0].h_rel is not None

        bboxes = Bboxes.concatenate([inst._bboxes for inst in instances_list], axis=axis)
        keypoints = (
            np.concatenate([inst.keypoints for inst in instances_list], axis=axis) if use_keypoint else None
        )
        
        # 处理番茄特有属性
        cluster_ids = np.concatenate([inst.cluster_ids for inst in instances_list if inst.cluster_ids is not None], axis=axis) \
            if any(inst.cluster_ids is not None for inst in instances_list) else None
        h_rel = np.concatenate([inst.h_rel for inst in instances_list if inst.h_rel is not None], axis=axis) \
            if any(inst.h_rel is not None for inst in instances_list) else None

        if use_segments:
            segments = []
            for inst in instances_list:
                if isinstance(inst.segments, np.ndarray):
                    segments.append(inst.segments)
                elif inst.segments:
                    segments.extend(inst.segments)
                else:
                    segments.extend([])
            if all(isinstance(x, np.ndarray) for x in segments):
                segments = np.concatenate(segments, axis=axis)
        else:
            segments = None

        return Instances(bboxes, segments, keypoints, bbox_format=instances_list[0].bbox_format, 
                         normalized=instances_list[0].normalized, cluster_ids=cluster_ids, h_rel=h_rel)

    def __getitem__(self, index) -> "Instances":
        """
        Returns a new Instances object containing only the instances specified by index.

        Args:
            index (int | slice | np.ndarray): Integer or slice index, or a boolean mask of length len(self).
        
        Returns:
            Instances: A new Instances object with the selected instances.
        """
        if isinstance(index, int):
            if index < 0:
                index = len(self) + index
            segments = None
            if self.segments is not None:
                if isinstance(self.segments, np.ndarray) and self.segments.size > 0:
                    segments = self.segments[index:index+1]
                elif isinstance(self.segments, list) and len(self.segments) > 0:
                    segments = [self.segments[index]]
            
            keypoints = None if self.keypoints is None else self.keypoints[index:index+1]
            
            # 处理番茄特有属性
            cluster_ids = None if self.cluster_ids is None else self.cluster_ids[index:index+1]
            h_rel = None if self.h_rel is None else self.h_rel[index:index+1]
            
            return Instances(
                self._bboxes.bboxes[index:index+1],
                segments,
                keypoints,
                self.normalized,
                self._bboxes.format,
                cluster_ids,
                h_rel,
            )
        
        # 处理切片或布尔索引
        segments = None
        if self.segments is not None:
            if isinstance(self.segments, np.ndarray) and self.segments.size > 0:
                segments = self.segments[index]
            elif isinstance(self.segments, list) and len(self.segments) > 0:
                segments = [self.segments[i] for i in index] if isinstance(index, list) else [self.segments[i] for i in range(len(self.segments)) if index[i]]
        
        keypoints = None if self.keypoints is None else self.keypoints[index]
        
        # 处理番茄特有属性
        cluster_ids = self.cluster_ids[index] if self.cluster_ids is not None else None
        h_rel = self.h_rel[index] if self.h_rel is not None else None
        
        return Instances(
            self._bboxes.bboxes[index],
            segments,
            keypoints,
            self.normalized,
            self._bboxes.format,
            cluster_ids,
            h_rel,
        )