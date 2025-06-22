#!/usr/bin/env python
"""
测试标签加载的简单脚本
"""

import os
import sys
from pathlib import Path

# 确保当前目录在Python路径中
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = str(Path(current_path).parent.parent)  # 修复路径
if parent_path not in sys.path:
    sys.path.append(parent_path)
if current_path not in sys.path:
    sys.path.append(current_path)

import yaml
import numpy as np

# 应用补丁
try:
    from ultralytics.utils.patches import apply_patches
    apply_patches()
    print("✓ 补丁已成功应用")
except Exception as e:
    print(f"× 应用补丁失败: {e}")

def test_data_loading():
    """测试数据加载功能"""
    # 获取数据路径 - 修复路径
    root_path = Path('/root/shared-nvme')
    data_path = root_path / "tomato_transStyle" / "data.yaml"
    
    if not data_path.exists():
        print(f"错误：数据文件不存在：{data_path}")
        return False
        
    # 加载数据配置
    with open(data_path, "r") as f:
        data = yaml.safe_load(f)
    
    print(f"数据配置: {data}")
    
    # 修复数据路径 - 绝对路径
    data_root = data_path.parent
    valid_images_path = data_root / "valid" / "images"
    print(f"验证集图像目录: {valid_images_path}")
    
    # 验证路径是否存在
    if not valid_images_path.exists():
        print(f"错误：验证集图像目录不存在: {valid_images_path}")
        return False
    
    # 导入必要的功能
    from ultralytics.data.utils import verify_image_label, exif_size
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.utils.torch_utils import model_info_for_loggers
    
    # 修改数据配置以使用绝对路径
    data_modified = data.copy()
    data_modified['val'] = str(valid_images_path) 
    
    # 创建完整的超参数
    class HyperParams:
        def __init__(self):
            # 基本变换参数
            self.mosaic = 0
            self.mixup = 0
            
            # Format类需要的参数
            self.mask_ratio = 4
            self.overlap_mask = True
            self.bgr = 0.0
            
            # 其他变换参数
            self.degrees = 0.0
            self.translate = 0.1
            self.scale = 0.5
            self.shear = 0.0
            self.perspective = 0.0
            self.copy_paste = 0.0
            self.copy_paste_mode = "flip"
            self.hsv_h = 0.015
            self.hsv_s = 0.7
            self.hsv_v = 0.4
            self.flipud = 0.0
            self.fliplr = 0.5
    
    hyp = HyperParams()
    
    print("初始化数据集...")
    try:
        # 初始化数据集
        dataset = YOLODataset(
            img_path=str(valid_images_path),
            imgsz=640,
            cache=False,
            augment=False,
            hyp=hyp,  # 提供hyp参数
            prefix="val: ",
            rect=True,
            batch_size=1,
            stride=32,
            pad=0.5,
            single_cls=False,
            data=data_modified
        )
    except Exception as e:
        print(f"初始化数据集出错: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 检查是否正确初始化
    print(f"数据集实例: {dataset}")
    print(f"数据集属性: has_cluster_id={dataset.has_cluster_id}, has_h_rel={dataset.has_h_rel}")
    
    # 获取标签
    print("开始加载标签...")
    try:
        labels = dataset.get_labels()
        if not labels:
            print("错误：未加载到标签")
            return False
    except Exception as e:
        print(f"获取标签出错: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    print(f"标签数量: {len(labels)}")
    first_label = labels[0]
    print(f"第一个标签信息: {first_label.keys()}")
    
    # 检查番茄特有字段
    has_cluster_id = any('cluster_ids' in label or 'cluster_id' in label for label in labels)
    has_h_rel = any('h_rel' in label for label in labels)
    
    print(f"cluster_id字段存在: {has_cluster_id}")
    print(f"h_rel字段存在: {has_h_rel}")
    
    # 输出几个示例标签
    print("\n示例标签:")
    for i, label in enumerate(labels[:3]):
        print(f"标签 {i+1}:")
        # 打印关键字段
        for key in ['im_file', 'shape', 'cls', 'bboxes', 'cluster_ids', 'h_rel']:
            if key in label:
                val = label[key]
                if isinstance(val, np.ndarray) and val.size > 10:
                    print(f"  {key}: {val.shape} {val[:2]}")
                else:
                    print(f"  {key}: {val}")
    
    # 尝试获取一个batch
    try:
        from torch.utils.data import DataLoader
        print("\n创建数据加载器...")
        loader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn, num_workers=0)
        
        print("\n尝试获取第一个批次...")
        for batch in loader:
            print("\n批次字段:", batch.keys())
            for k, v in batch.items():
                if hasattr(v, 'shape'):
                    print(f"  {k}: {v.shape}")
                else:
                    print(f"  {k}: {type(v)}")
            break  # 只获取第一个批次
    except Exception as e:
        print(f"加载batch出错: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("\n===== 开始测试标签加载 =====\n")
    success = test_data_loading()
    print("\n===== 测试结束 =====")
    print(f"结果: {'成功' if success else '失败'}") 