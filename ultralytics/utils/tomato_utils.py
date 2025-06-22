# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from tqdm import tqdm


def generate_cluster_labels(dataset_path, save_path=None, cluster_method="dbscan", 
                           distance_threshold=0.2, min_samples=2, plot=False):
    """
    为番茄数据集生成果串聚类标签。
    
    参数:
        dataset_path (str): 包含images和labels文件夹的数据集路径
        save_path (str, optional): 保存新标签的路径。如果为None，则覆盖原标签
        cluster_method (str): 聚类方法，目前支持"dbscan"
        distance_threshold (float): DBSCAN的eps参数，定义邻域距离
        min_samples (int): DBSCAN的最小样本数
        plot (bool): 是否生成聚类可视化图像
        
    返回:
        None
    """
    # 设置路径
    dataset_path = Path(dataset_path)
    labels_dir = dataset_path / "labels"
    images_dir = dataset_path / "images"
    
    if save_path is not None:
        save_path = Path(save_path)
        os.makedirs(save_path / "labels", exist_ok=True)
        os.makedirs(save_path / "images", exist_ok=True)
    
    # 检查目录是否存在
    if not labels_dir.exists():
        raise FileNotFoundError(f"标签目录不存在: {labels_dir}")
    
    # 处理每个标签文件
    label_files = list(labels_dir.glob("*.txt"))
    print(f"找到 {len(label_files)} 个标签文件")
    
    for label_file in tqdm(label_files, desc="处理番茄果串聚类"):
        # 读取标签
        with open(label_file, "r") as f:
            lines = f.readlines()
        
        # 解析标签
        boxes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:  # 确保至少有class x y w h
                continue
                
            cls_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # 只对单果类别(0-3)进行聚类分析，跳过果串类别(4-5)
            if cls_id < 4:  # 单果类别
                boxes.append([cls_id, x_center, y_center, width, height])
        
        if not boxes:
            continue
            
        boxes = np.array(boxes)
        
        # 提取坐标进行聚类
        positions = boxes[:, 1:3]  # x_center, y_center
        
        # 进行聚类
        if cluster_method == "dbscan":
            # 使用DBSCAN进行聚类，主要基于x坐标相近度
            x_weight = 3.0  # x坐标权重较大，因为同一果串在垂直方向上变化较大但水平位置接近
            weighted_positions = positions.copy()
            weighted_positions[:, 0] *= x_weight  # 增加x坐标权重
            
            clustering = DBSCAN(eps=distance_threshold, min_samples=min_samples).fit(weighted_positions)
            cluster_ids = clustering.labels_
        else:
            raise ValueError(f"不支持的聚类方法: {cluster_method}")
        
        # 为每个聚类计算h_rel (相对高度)
        new_labels = []
        for cluster_id in np.unique(cluster_ids):
            if cluster_id < 0:  # 噪声点
                # 为噪声点分配唯一ID并设置h_rel为0.5
                for i, (box, cid) in enumerate(zip(boxes, cluster_ids)):
                    if cid == cluster_id:
                        h_rel = 0.5  # 默认值
                        new_labels.append((box, -i-1000, h_rel))  # 负数作为噪声点的唯一ID
            else:
                # 获取属于当前聚类的所有box
                cluster_boxes = boxes[cluster_ids == cluster_id]
                cluster_y = cluster_boxes[:, 2]  # y坐标
                
                # 计算y的最小值和最大值，用于归一化
                y_min, y_max = np.min(cluster_y), np.max(cluster_y)
                
                # 对于同一聚类中的每个box
                for box, y in zip(cluster_boxes, cluster_y):
                    # 计算相对高度 (0=顶部，1=底部)
                    if y_max > y_min:
                        h_rel = (y - y_min) / (y_max - y_min) 
                    else:
                        h_rel = 0.5
                        
                    new_labels.append((box, cluster_id, h_rel))
                    
                # 为每个聚类添加一个果串标签
                # 根据聚类中番茄的成熟度判断果串类别
                cluster_cls = cluster_boxes[:, 0]
                # 如果绝大多数(>60%)是成熟的(类别0-1)，则标记为成熟果串
                # 注意：0-Fully_Ripe, 1-Ripe是成熟的类别
                ripe_ratio = np.sum((cluster_cls <= 1)) / len(cluster_cls)
                if ripe_ratio > 0.6:
                    bunch_cls = 4  # Ripe_Bunch
                else:
                    bunch_cls = 5  # Unripe_Bunch
                
                # 计算果串的边界框（包含所有单果）
                x_min, x_max = np.min(cluster_boxes[:, 1] - cluster_boxes[:, 3]/2), np.max(cluster_boxes[:, 1] + cluster_boxes[:, 3]/2)
                y_min, y_max = np.min(cluster_boxes[:, 2] - cluster_boxes[:, 4]/2), np.max(cluster_boxes[:, 2] + cluster_boxes[:, 4]/2)
                
                # 计算果串边界框的中心点和尺寸
                bunch_x = (x_min + x_max) / 2
                bunch_y = (y_min + y_max) / 2
                bunch_w = x_max - x_min
                bunch_h = y_max - y_min
                
                # 稍微扩大果串边界框
                bunch_w *= 1.1
                bunch_h *= 1.1
                
                # 添加果串标签，h_rel设为-1表示无意义
                bunch_box = [bunch_cls, bunch_x, bunch_y, bunch_w, bunch_h]
                new_labels.append((bunch_box, cluster_id, -1))
        
        # 生成新标签文件
        output_lines = []
        for (box, cluster_id, h_rel) in new_labels:
            # 格式: class_id x_center y_center width height cluster_id h_rel
            output_line = f"{int(box[0])} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f} {int(cluster_id)} {h_rel:.6f}\n"
            output_lines.append(output_line)
        
        # 保存新标签
        if save_path is not None:
            output_file = save_path / "labels" / label_file.name
        else:
            output_file = label_file
            
        with open(output_file, "w") as f:
            f.writelines(output_lines)
            
        # 可视化聚类结果
        if plot:
            visualize_clusters(label_file.stem, boxes, cluster_ids, new_labels, 
                              dataset_path, save_path or dataset_path)


def visualize_clusters(image_name, boxes, cluster_ids, new_labels, dataset_path, save_path):
    """可视化聚类结果"""
    plt.figure(figsize=(10, 8))
    plt.title(f"番茄果串聚类 - {image_name}")
    
    # 准备颜色
    unique_clusters = np.unique(cluster_ids)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
    color_map = {cid: colors[i] for i, cid in enumerate(unique_clusters)}
    
    # 类别名称映射
    class_names = {
        0: "Fully_Ripe",
        1: "Ripe", 
        2: "Breaking",
        3: "Green",
        4: "Ripe_Bunch",
        5: "Unripe_Bunch"
    }
    
    # 绘制每个框和聚类
    for (box, cluster_id, h_rel) in new_labels:
        x, y = box[1], box[2]
        w, h = box[3], box[4]
        cls = int(box[0])
        
        # 获取聚类颜色
        color = color_map.get(cluster_id, 'black')
        
        # 绘制边界框
        rect = plt.Rectangle((x-w/2, y-h/2), w, h, fill=False, 
                           edgecolor=color, linewidth=2)
        plt.gca().add_patch(rect)
        
        # 添加标签：类别名称、聚类ID和相对高度
        cls_name = class_names.get(cls, f"Class{cls}")
        label_text = f"{cls_name} C:{cluster_id} H:{h_rel:.2f}"
        
        # 果串类别(4-5)使用更大的字体和不同的风格
        if cls >= 4:
            plt.text(x, y, label_text, 
                    color='black', fontsize=10, fontweight='bold',
                    bbox=dict(facecolor=color, alpha=0.7, boxstyle='round'))
        else:
            plt.text(x, y, label_text, 
                    color='white', fontsize=8,
                    bbox=dict(facecolor=color, alpha=0.7))
    
    # 设置坐标轴
    plt.xlim(0, 1)
    plt.ylim(1, 0)  # 翻转y轴，使图像坐标与通常的图像显示一致
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    
    # 保存图像
    vis_dir = save_path / "visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    plt.savefig(vis_dir / f"{image_name}_clusters.png")
    plt.close()


def generate_bunch_labels(input_file, output_file):
    """
    根据单果标签生成果串标签。
    
    参数:
        input_file (str): 输入标签文件路径
        output_file (str): 输出标签文件路径
    """
    # 读取原始标签
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # 解析为对象列表
    tomatoes = []
    clusters = {}
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 7:  # class_id x y w h cluster_id h_rel
            cls_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            cluster_id = int(parts[5])
            h_rel = float(parts[6])
            
            # 只处理单果类别(0-3)
            if cls_id < 4:
                tomatoes.append((cls_id, x, y, w, h, cluster_id, h_rel))
                
                # 按cluster_id分组
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append((cls_id, x, y, w, h))
    
    # 为每个聚类生成果串标签
    bunch_labels = []
    
    for cluster_id, tomato_list in clusters.items():
        if len(tomato_list) < 2:  # 至少需要2个番茄才能构成一个果串
            continue
        
        # 提取类别，计算成熟度比例
        classes = [t[0] for t in tomato_list]
        # 注意：0-Fully_Ripe, 1-Ripe是成熟的类别
        ripe_ratio = sum(1 for c in classes if c <= 1) / len(classes)
        
        # 决定果串类别
        if ripe_ratio >= 0.6:  # 如果60%以上是成熟的，则为成熟果串
            bunch_cls = 4  # Ripe_Bunch
        else:
            bunch_cls = 5  # Unripe_Bunch
            
        # 计算果串的边界框
        x_coords = [t[1] for t in tomato_list]
        y_coords = [t[2] for t in tomato_list]
        w_half = [t[3]/2 for t in tomato_list]
        h_half = [t[4]/2 for t in tomato_list]
        
        x_min = min([x - w for x, w in zip(x_coords, w_half)])
        x_max = max([x + w for x, w in zip(x_coords, w_half)])
        y_min = min([y - h for y, h in zip(y_coords, h_half)])
        y_max = max([y + h for y, h in zip(y_coords, h_half)])
        
        # 计算中心点和宽高
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = (x_max - x_min) * 1.1  # 稍微扩大边界框
        h = (y_max - y_min) * 1.1
        
        # 添加果串标签，h_rel设为-1表示无意义
        bunch_labels.append((bunch_cls, cx, cy, w, h, cluster_id, -1))
    
    # 组合单果和果串标签
    output_lines = []
    
    # 添加原始单果标签
    for line in lines:
        output_lines.append(line)
    
    # 添加果串标签
    for (cls, x, y, w, h, cluster_id, h_rel) in bunch_labels:
        line = f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {cluster_id} {h_rel:.6f}\n"
        output_lines.append(line)
    
    # 保存结果
    with open(output_file, 'w') as f:
        f.writelines(output_lines)


def format_yolo_labels(input_file, output_file, add_cluster=True, add_rel_height=True):
    """
    将常规YOLO标签转换为包含cluster_id和h_rel的格式。
    
    参数:
        input_file (str): 输入标签文件路径
        output_file (str): 输出标签文件路径
        add_cluster (bool): 是否添加聚类ID
        add_rel_height (bool): 是否添加相对高度
    """
    # 读取原始标签
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # 解析为对象列表
    objects = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            cls_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            objects.append((cls_id, x, y, w, h))
    
    # 进行简单聚类 (基于x坐标的邻近度)
    if add_cluster:
        # 只对单果类别(0-3)进行聚类
        fruit_objects = [(i, obj) for i, obj in enumerate(objects) if obj[0] < 4]
        if fruit_objects:
            indices, fruit_objs = zip(*fruit_objects)
            # 仅使用x坐标进行聚类
            x_coords = np.array([obj[1] for obj in fruit_objs]).reshape(-1, 1)
            if len(x_coords) > 1:
                clustering = DBSCAN(eps=0.1, min_samples=1).fit(x_coords)
                fruit_cluster_ids = clustering.labels_
                
                # 为所有对象分配聚类ID
                cluster_ids = np.zeros(len(objects), dtype=int)
                for i, (idx, cluster_id) in enumerate(zip(indices, fruit_cluster_ids)):
                    cluster_ids[idx] = cluster_id
            else:
                cluster_ids = np.zeros(len(objects), dtype=int)
        else:
            cluster_ids = np.zeros(len(objects), dtype=int)
    else:
        cluster_ids = np.zeros(len(objects), dtype=int)
    
    # 为每个聚类分配h_rel
    if add_rel_height:
        # 按聚类ID分组计算相对高度
        result = []
        for cluster_id in np.unique(cluster_ids):
            # 获取此聚类的索引
            indices = np.where(cluster_ids == cluster_id)[0]
            # 获取y坐标和类别
            classes = np.array([objects[i][0] for i in indices])
            
            # 只对单果类别(0-3)计算相对高度
            fruit_indices = [i for i in indices if objects[i][0] < 4]
            
            if fruit_indices:
                y_coords = np.array([objects[i][2] for i in fruit_indices])
                if len(y_coords) > 1:
                    y_min, y_max = np.min(y_coords), np.max(y_coords)
                    for i in indices:
                        obj = objects[i]
                        # 计算相对高度 (0=顶部，1=底部)
                        if obj[0] < 4:  # 单果
                            h_rel = (obj[2] - y_min) / (y_max - y_min) if y_max > y_min else 0.5
                        else:  # 果串类别(4-5)的h_rel设为-1表示无意义
                            h_rel = -1
                        result.append((obj[0], obj[1], obj[2], obj[3], obj[4], cluster_id, h_rel))
                else:
                    # 单个对象，单果相对高度设为0.5，果串设为-1
                    for i in indices:
                        obj = objects[i]
                        h_rel = 0.5 if obj[0] < 4 else -1  # 单果0.5，果串-1
                        result.append((obj[0], obj[1], obj[2], obj[3], obj[4], cluster_id, h_rel))
            else:
                # 没有单果，全部是果串，h_rel设为-1
                for i in indices:
                    obj = objects[i]
                    result.append((obj[0], obj[1], obj[2], obj[3], obj[4], cluster_id, -1))
    else:
        # 不添加相对高度，只添加聚类ID
        result = [(obj[0], obj[1], obj[2], obj[3], obj[4], cid, -1 if obj[0] >= 4 else 0.5) 
                 for obj, cid in zip(objects, cluster_ids)]
    
    # 写入结果
    with open(output_file, 'w') as f:
        for obj in result:
            if add_cluster and add_rel_height:
                f.write(f"{int(obj[0])} {obj[1]:.6f} {obj[2]:.6f} {obj[3]:.6f} {obj[4]:.6f} {int(obj[5])} {obj[6]:.6f}\n")
            elif add_cluster:
                f.write(f"{int(obj[0])} {obj[1]:.6f} {obj[2]:.6f} {obj[3]:.6f} {obj[4]:.6f} {int(obj[5])}\n")
            else:
                f.write(f"{int(obj[0])} {obj[1]:.6f} {obj[2]:.6f} {obj[3]:.6f} {obj[4]:.6f}\n")


def batch_format_dataset(input_dir, output_dir=None, add_cluster=True, add_rel_height=True):
    """
    批量处理整个数据集的标签。
    
    参数:
        input_dir (str): 输入数据集目录 (包含images和labels子目录)
        output_dir (str): 输出数据集目录，如果为None则覆盖原始文件
        add_cluster (bool): 是否添加聚类ID
        add_rel_height (bool): 是否添加相对高度
    """
    input_dir = Path(input_dir)
    if output_dir is not None:
        output_dir = Path(output_dir)
        os.makedirs(output_dir / "labels", exist_ok=True)
        os.makedirs(output_dir / "images", exist_ok=True)
    
    # 处理labels目录
    labels_dir = input_dir / "labels"
    if not labels_dir.exists():
        raise FileNotFoundError(f"标签目录不存在: {labels_dir}")
    
    # 获取所有txt文件
    label_files = list(labels_dir.glob("*.txt"))
    print(f"找到 {len(label_files)} 个标签文件")
    
    # 处理每个文件
    for input_file in tqdm(label_files, desc="处理标签"):
        output_file = output_dir / "labels" / input_file.name if output_dir else input_file
        format_yolo_labels(input_file, output_file, add_cluster, add_rel_height)
    
    # 如果指定了输出目录，复制图像
    if output_dir is not None:
        images_dir = input_dir / "images"
        if images_dir.exists():
            from shutil import copy2
            for img_file in tqdm(list(images_dir.glob("*.*")), desc="复制图像"):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    copy2(img_file, output_dir / "images" / img_file.name)
                    
    print(f"处理完成。输出目录：{output_dir or input_dir}") 