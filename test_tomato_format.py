import sys
import os
import logging
import torch
import numpy as np
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestTomatoFormat")

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入必要的类
from ultralytics.data.augment import TomatoFormat
from ultralytics.utils.instance import Instances

def create_mock_data():
    """创建模拟数据用于测试"""
    # 创建一个简单的图像
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    img[100:200, 100:200, 0] = 255  # 添加一个红色方块
    
    # 创建边界框 [x, y, w, h]
    bboxes = np.array([[150, 150, 100, 100], [300, 300, 50, 50]], dtype=np.float32)
    
    # 创建类别标签
    cls = np.array([0, 1], dtype=np.int64)
    
    # 创建cluster_ids
    cluster_ids = np.array([[1], [2]], dtype=np.float32)
    
    # 创建h_rel
    h_rel = np.array([[0.5], [0.7]], dtype=np.float32)
    
    # 创建Instances对象
    instances = Instances(bboxes=bboxes, normalized=False)
    instances.cluster_ids = cluster_ids
    instances.h_rel = h_rel
    
    # 创建标签字典
    labels = {
        "img": img,
        "cls": cls,
        "instances": instances
    }
    
    return labels

def test_tomato_format():
    """测试TomatoFormat类"""
    logger.info("开始测试TomatoFormat类")
    
    # 创建模拟数据
    labels = create_mock_data()
    logger.info(f"创建模拟数据成功: img.shape={labels['img'].shape}, cls.shape={labels['cls'].shape}")
    
    # 创建TomatoFormat对象
    formatter = TomatoFormat(bbox_format="xywh", normalize=True)
    logger.info("创建TomatoFormat对象成功")
    
    try:
        # 应用格式化
        logger.info("开始应用TomatoFormat格式化...")
        formatted_labels = formatter(labels)
        logger.info("格式化成功！")
        
        # 打印格式化后的标签
        logger.info("格式化后的标签:")
        for k, v in formatted_labels.items():
            if isinstance(v, torch.Tensor):
                logger.info(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            else:
                logger.info(f"  {k}: type={type(v)}")
        
        return True
    except Exception as e:
        logger.error(f"格式化过程中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_tomato_format()
    if success:
        logger.info("测试成功完成！")
    else:
        logger.error("测试失败！") 