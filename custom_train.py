import argparse
import sys
import os
import glob
from pathlib import Path

# Add project root to sys.path to ensure ultralytics can be imported
# This assumes train.py is in the project root
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

# 在训练前确保软链接已经创建
def ensure_symlinks():
    # 训练集
    train_images_dir = "/root/shared-nvme/tomato_transStyle/train/images"
    train_labels_dir = "/root/shared-nvme/tomato_transStyle/train/labels"
    train_symlink = os.path.join(train_images_dir, "labels")
    
    # 验证集
    valid_images_dir = "/root/shared-nvme/tomato_transStyle/valid/images"
    valid_labels_dir = "/root/shared-nvme/tomato_transStyle/valid/labels"
    valid_symlink = os.path.join(valid_images_dir, "labels")
    
    # 测试集
    test_images_dir = "/root/shared-nvme/tomato_transStyle/test/images"
    test_labels_dir = "/root/shared-nvme/tomato_transStyle/test/labels"
    test_symlink = os.path.join(test_images_dir, "labels")
    
    # 创建训练集软链接
    if not os.path.exists(train_symlink):
        print(f"创建训练集软链接: {train_symlink} -> {train_labels_dir}")
        os.symlink(train_labels_dir, train_symlink)
    
    # 创建验证集软链接
    if not os.path.exists(valid_symlink):
        print(f"创建验证集软链接: {valid_symlink} -> {valid_labels_dir}")
        os.symlink(valid_labels_dir, valid_symlink)
    
    # 创建测试集软链接
    if not os.path.exists(test_symlink):
        print(f"创建测试集软链接: {test_symlink} -> {test_labels_dir}")
        os.symlink(test_labels_dir, test_symlink)

# 检查数据集标签
def check_dataset_labels():
    # 检查验证集
    valid_images_dir = "/root/shared-nvme/tomato_transStyle/valid/images"
    valid_labels_dir = "/root/shared-nvme/tomato_transStyle/valid/labels"
    
    # 获取验证集中的所有图像文件
    valid_images = glob.glob(os.path.join(valid_images_dir, "*.jpg"))
    print(f"找到 {len(valid_images)} 张验证集图像")
    
    # 检查前5个图像是否有对应的标签文件
    print("\n检查验证集前5个图像的标签文件:")
    label_count = 0
    for i, img_path in enumerate(valid_images[:5]):
        img_name = os.path.basename(img_path)
        label_name = img_name.replace(".jpg", ".txt")
        label_path = os.path.join(valid_labels_dir, label_name)
        symlink_label_path = os.path.join(valid_images_dir, "labels", label_name)
        
        print(f"{i+1}. 图像: {img_name}")
        print(f"   原始标签路径: {label_path} (存在: {os.path.exists(label_path)})")
        print(f"   软链接标签路径: {symlink_label_path} (存在: {os.path.exists(symlink_label_path)})")
        
        # 如果标签文件存在，显示其内容
        if os.path.exists(label_path):
            label_count += 1
            with open(label_path, 'r') as f:
                lines = f.readlines()
                print(f"   标签行数: {len(lines)}")
                if lines:
                    print(f"   第一行内容: {lines[0].strip()}")
        print()
    
    print(f"验证集中找到 {label_count}/5 个标签文件")
    
    # 检查训练集
    train_images_dir = "/root/shared-nvme/tomato_transStyle/train/images"
    train_labels_dir = "/root/shared-nvme/tomato_transStyle/train/labels"
    
    # 获取训练集中的所有图像文件
    train_images = glob.glob(os.path.join(train_images_dir, "*.jpg"))
    print(f"找到 {len(train_images)} 张训练集图像")
    
    # 检查前5个图像是否有对应的标签文件
    print("\n检查训练集前5个图像的标签文件:")
    label_count = 0
    for i, img_path in enumerate(train_images[:5]):
        img_name = os.path.basename(img_path)
        label_name = img_name.replace(".jpg", ".txt")
        label_path = os.path.join(train_labels_dir, label_name)
        symlink_label_path = os.path.join(train_images_dir, "labels", label_name)
        
        print(f"{i+1}. 图像: {img_name}")
        print(f"   原始标签路径: {label_path} (存在: {os.path.exists(label_path)})")
        print(f"   软链接标签路径: {symlink_label_path} (存在: {os.path.exists(symlink_label_path)})")
        
        # 如果标签文件存在，显示其内容
        if os.path.exists(label_path):
            label_count += 1
            with open(label_path, 'r') as f:
                lines = f.readlines()
                print(f"   标签行数: {len(lines)}")
                if lines:
                    print(f"   第一行内容: {lines[0].strip()}")
                    parts = lines[0].strip().split()
                    print(f"   标签列数: {len(parts)}")
                    if len(parts) >= 7:
                        print(f"   扩展属性: cluster_id={parts[5]}, h_rel={parts[6]}")
        print()
    
    print(f"训练集中找到 {label_count}/5 个标签文件")

# 创建自定义数据集YAML文件
def create_custom_yaml():
    yaml_path = "/root/shared-nvme/tomato_transStyle/custom_data.yaml"
    
    yaml_content = """
path: /root/shared-nvme/tomato_transStyle

train: train/images
val: valid/images
test: test/images

nc: 6
names: ['fully ripe', 'ripe', 'turning', 'green', 'ripe bunch', 'unripe bunch']
has_cluster_id: true
has_h_rel: true
"""
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"创建自定义YAML文件: {yaml_path}")
    return yaml_path

if __name__ == '__main__':
    print("=" * 50)
    print("开始番茄检测模型训练")
    print("=" * 50)
    
    # 确保软链接存在
    ensure_symlinks()
    
    # 检查数据集标签
    check_dataset_labels()
    
    # 创建自定义YAML文件
    yaml_path = create_custom_yaml()
    
    # 初始化模型
    print("加载模型...")
    model_path = r'/root/shared-nvme/yolov12-main/yolov12-new/ultralytics/cfg/models/v12/yolo12s-A2C2f-DYT-EfficientHead-MambaOut.yaml'
    print(f"模型路径: {model_path}")
    print(f"模型文件存在: {os.path.exists(model_path)}")
    
    model = YOLO(model_path)
    print("模型加载完成")
    
    # 开始训练
    print("开始训练...")
    model.train(data=yaml_path,
                imgsz=640,
                epochs=50,  # 减少训练轮数
                batch=4,    # 使用较小的batch size
                workers=2,  # 使用较少的worker
                optimizer='SGD',
                close_mosaic=10,
                project='runs/train',
                name='yolov12_custom_exp',
                single_cls=False,
                lr0=0.01,
                cache=False,
                loss='TomatoDetectWithRankLoss(lambda_pos=0.3, lambda_rank=0.2, lambda_conf=0.4)',
                hsv_h=0.0,  # 关闭HSV色调增强
                hsv_s=0.0,  # 关闭HSV饱和度增强
                hsv_v=0.0,  # 关闭HSV亮度增强
                degrees=0.0,  # 关闭旋转
                translate=0.0,  # 关闭平移
                scale=0.0,  # 关闭缩放
                shear=0.0,  # 关闭剪切
                perspective=0.0,  # 关闭透视变换
                flipud=0.0,  # 关闭上下翻转
                fliplr=0.0,  # 关闭左右翻转
                mosaic=0.0,  # 关闭拼接
                mixup=0.0,  # 关闭混合
                verbose=True  # 开启详细日志
                ) 