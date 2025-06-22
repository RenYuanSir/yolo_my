# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Monkey patches to update/extend functionality of existing functions."""

import time
from pathlib import Path

import cv2
import numpy as np
import torch

# OpenCV Multilanguage-friendly functions ------------------------------------------------------------------------------
_imshow = cv2.imshow  # copy to avoid recursion errors


def imread(filename: str, flags: int = cv2.IMREAD_COLOR):
    """
    Read an image from a file.

    Args:
        filename (str): Path to the file to read.
        flags (int, optional): Flag that can take values of cv2.IMREAD_*. Defaults to cv2.IMREAD_COLOR.

    Returns:
        (np.ndarray): The read image.
    """
    return cv2.imdecode(np.fromfile(filename, np.uint8), flags)


def imwrite(filename: str, img: np.ndarray, params=None):
    """
    Write an image to a file.

    Args:
        filename (str): Path to the file to write.
        img (np.ndarray): Image to write.
        params (list of ints, optional): Additional parameters. See OpenCV documentation.

    Returns:
        (bool): True if the file was written, False otherwise.
    """
    try:
        cv2.imencode(Path(filename).suffix, img, params)[1].tofile(filename)
        return True
    except Exception:
        return False


def imshow(winname: str, mat: np.ndarray):
    """
    Displays an image in the specified window.

    Args:
        winname (str): Name of the window.
        mat (np.ndarray): Image to be shown.
    """
    _imshow(winname.encode("unicode_escape").decode(), mat)


# PyTorch functions ----------------------------------------------------------------------------------------------------
_torch_load = torch.load  # copy to avoid recursion errors
_torch_save = torch.save


def torch_load(*args, **kwargs):
    """
    Load a PyTorch model with updated arguments to avoid warnings.

    This function wraps torch.load and adds the 'weights_only' argument for PyTorch 1.13.0+ to prevent warnings.

    Args:
        *args (Any): Variable length argument list to pass to torch.load.
        **kwargs (Any): Arbitrary keyword arguments to pass to torch.load.

    Returns:
        (Any): The loaded PyTorch object.

    Note:
        For PyTorch versions 2.0 and above, this function automatically sets 'weights_only=False'
        if the argument is not provided, to avoid deprecation warnings.
    """
    from ultralytics.utils.torch_utils import TORCH_1_13

    if TORCH_1_13 and "weights_only" not in kwargs:
        kwargs["weights_only"] = False

    return _torch_load(*args, **kwargs)


def torch_save(*args, **kwargs):
    """
    Optionally use dill to serialize lambda functions where pickle does not, adding robustness with 3 retries and
    exponential standoff in case of save failure.

    Args:
        *args (tuple): Positional arguments to pass to torch.save.
        **kwargs (Any): Keyword arguments to pass to torch.save.
    """
    for i in range(4):  # 3 retries
        try:
            return _torch_save(*args, **kwargs)
        except RuntimeError as e:  # unable to save, possibly waiting for device to flush or antivirus scan
            if i == 3:
                raise e
            time.sleep((2**i) / 2)  # exponential standoff: 0.5s, 1.0s, 2.0s


"""
补丁文件，用于修复数据集加载和验证过程中的问题
"""

import os
import numpy as np
import torch
from pathlib import Path
from ultralytics.utils import LOGGER, colorstr

def patch_verify_image_label(args):
    """
    修补后的verify_image_label函数，增强对番茄数据集特殊属性的处理能力
    
    Args:
        args: 与原函数相同的参数
        
    Returns:
        与原函数相同的返回值，但确保能正确处理番茄数据集特殊属性
    """
    # 解包参数
    im_file, lb_file, prefix, keypoint_items, return_tuple, use_keypoints, use_segments, use_obb, has_cluster_id, has_h_rel = args
    
    try:
        # 验证图像
        im = verify_image(im_file)
        shape = im.shape
        assert (shape[0] > 9) & (shape[1] > 9), f"Image size {shape} < 10 pixels"
        assert im.shape[2] == 3, f"Image {im_file} isn't 3-channel (RGB)"
        assert lb_file.exists(), f"Label file {lb_file} not found"
        
        # 验证标签内容
        with open(lb_file) as f:
            lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
            if any(len(x) > 6 + keypoint_items * 3 for x in lb):  # 检查是否有超过标准格式+属性的行
                lb = [x for x in lb if len(x) >= 5 + (has_cluster_id + has_h_rel)]  # 过滤掉格式不正确的行
        
        if len(lb):
            # 标准的YOLO标签格式总是以类别开始，后跟bbox坐标
            classes = np.array([x[0] for x in lb], dtype=np.float32)
            
            # 转换标签行为numpy数组，处理不同的标签格式
            labels = []
            segments = []
            keypoints = []
            
            for x in lb:
                # 确保至少有5个元素（1个类别 + 4个bbox坐标）
                if len(x) >= 5:
                    # 创建标准标签数组：类别 + bbox坐标
                    label = np.zeros(5 + int(has_cluster_id) + int(has_h_rel), dtype=np.float32)
                    label[0] = float(x[0])  # 类别
                    label[1:5] = np.array(x[1:5], dtype=np.float32)  # bbox坐标
                    
                    # 添加cluster_id和h_rel，如果存在
                    if has_cluster_id and len(x) >= 6:
                        label[5] = float(x[5])
                    if has_h_rel and len(x) >= 7:
                        label[6] = float(x[6])
                        
                    labels.append(label)
                    
                    # 处理分割数据和关键点数据
                    # 暂时跳过分割和关键点处理
            
            if len(labels):
                labels = np.array(labels, dtype=np.float32)
                # 验证类别
                assert all(labels[:, 0] < 100), f"Label class {labels[:, 0]} exceeds dataset class count"
                
                # 验证bbox坐标
                assert (labels[:, 1:5] <= 1.0).all(), f"Non-normalized or out of bounds coordinates {labels[:, 1:5]}"
                assert (labels[:, 1:5] >= 0.0).all(), f"Non-normalized or out of bounds coordinates {labels[:, 1:5]}"
                
                # 如果只需要bbox格式，直接返回
                if not use_segments and not use_keypoints and not use_obb:
                    if return_tuple:
                        return im_file, labels, shape, segments, keypoints, None, None, None
                    else:
                        return im_file, labels, shape, segments, keypoints
                
                # 处理段数据 - 暂时跳过
                
                # 处理关键点数据 - 暂时跳过
            
                if return_tuple:
                    return im_file, labels, shape, segments, keypoints, None, None, None
                else:
                    return im_file, labels, shape, segments, keypoints
            else:
                # 没有有效标签，返回空数据
                labels = np.zeros((0, 5 + int(has_cluster_id) + int(has_h_rel)), dtype=np.float32)
                segments = []
                keypoints = np.zeros((0, keypoint_items, 3), dtype=np.float32) if keypoint_items else None
                
                if return_tuple:
                    return im_file, labels, shape, segments, keypoints, 0, 1, 0
                else:
                    return im_file, labels, shape, segments, keypoints
        else:
            # 文件为空
            labels = np.zeros((0, 5 + int(has_cluster_id) + int(has_h_rel)), dtype=np.float32)
            segments = []
            keypoints = np.zeros((0, keypoint_items, 3), dtype=np.float32) if keypoint_items else None
            
            if return_tuple:
                return im_file, labels, shape, segments, keypoints, 0, 1, 0
            else:
                return im_file, labels, shape, segments, keypoints
    
    except Exception as e:
        # 捕获任何异常
        LOGGER.warning(f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}")
        if return_tuple:
            return None, None, None, None, None, None, None, None
        else:
            return None, None, None, None, None


def verify_image(im_file):
    """基于原始verify_image函数的功能实现"""
    import cv2
    try:
        im = cv2.imread(str(im_file))
        return im
    except Exception:
        return None


def apply_patches():
    """
    应用所有补丁
    """
    try:
        # 应用原始补丁
        # Apply cv2 patches for non-ASCII and non-UTF characters in image paths
        import cv2
        if os.name == 'nt':  # Windows
            cv2.imread, cv2.imwrite, cv2.imshow = imread, imwrite, imshow
        
        # 应用torch补丁
        torch.load = torch_load
        torch.save = torch_save
        
        # 导入原始函数
        from ultralytics.data.utils import verify_image_label as original_verify_image_label
        
        # 保存原始函数的引用
        if not hasattr(original_verify_image_label, "_original"):
            original_verify_image_label._original = original_verify_image_label
            
        # 替换为补丁函数
        import ultralytics.data.utils
        ultralytics.data.utils.verify_image_label = patch_verify_image_label
        
        LOGGER.info(f"{colorstr('patches:')} 成功应用verify_image_label补丁")
        return True
    except Exception as e:
        LOGGER.warning(f"{colorstr('patches:')} 应用补丁时出错: {e}")
        return False
