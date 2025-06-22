import os
import cv2
import glob
from pathlib import Path


def process_images_with_yolo(model, input_img_dir, existing_label_dir, output_label_dir, conf_thresh=0.5):
    """
    使用YOLO模型处理图片并合并标注

    参数:
        model: 加载的YOLO模型
        input_img_dir: 输入图片目录
        existing_label_dir: 已有标注文件目录
        output_label_dir: 输出标注文件目录
        conf_thresh: 置信度阈值 (默认0.5)
    """
    # 确保输出目录存在
    os.makedirs(output_label_dir, exist_ok=True)

    # 获取所有图片文件
    img_files = glob.glob(os.path.join(input_img_dir, "*.jpg")) + \
                glob.glob(os.path.join(input_img_dir, "*.png")) + \
                glob.glob(os.path.join(input_img_dir, "*.jpeg"))

    for img_path in img_files:
        # 获取对应的标注文件路径
        img_name = Path(img_path).stem
        label_path = os.path.join(existing_label_dir, f"{img_name}.txt")
        output_path = os.path.join(output_label_dir, f"{img_name}.txt")

        # 读取现有标注（如果有）
        existing_annotations = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                existing_annotations = [line.strip() for line in f.readlines() if line.strip()]

        # 使用YOLO模型检测图片
        img = cv2.imread(img_path)
        results = model(img)

        # 解析检测结果
        new_annotations = []
        for result in results:
            for box in result.boxes:
                if box.conf.item() >= conf_thresh:  # 过滤低置信度检测
                    # 获取YOLO格式的标注 (class, x_center, y_center, width, height)
                    class_id = int(box.cls.item())
                    xywh = box.xywhn.cpu().numpy()[0]  # 归一化坐标
                    x_center, y_center, w, h = xywh

                    # 格式化为YOLO标注字符串
                    annotation = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
                    new_annotations.append(annotation)

        # 合并新旧标注
        merged_annotations = existing_annotations + new_annotations

        # 写入输出文件
        with open(output_path, 'w') as f:
            for ann in merged_annotations:
                f.write(ann + '\n')

        print(f"Processed: {img_name} - {len(new_annotations)} new annotations added")


if __name__ == "__main__":
    from ultralytics import YOLO

    # 配置参数
    MODEL_PATH = r"D:\yoloProject\yolov12-main\yolo_detection_system\assets\models\best-2025-5.14.pt"  # 你的YOLO模型路径
    IMAGE_DIR = r"D:\TomatoDataset\Tomoto Cluster\images"  # 图片目录
    EXISTING_LABEL_DIR = r"D:\TomatoDataset\Tomoto Cluster\labels_converted"  # 现有标注目录
    OUTPUT_LABEL_DIR = r"D:\TomatoDataset\Tomoto Cluster\labels_anno"  # 输出标注目录

    # 加载模型
    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    # 处理图片
    print("Processing images...")
    process_images_with_yolo(
        model=model,
        input_img_dir=IMAGE_DIR,
        existing_label_dir=EXISTING_LABEL_DIR,
        output_label_dir=OUTPUT_LABEL_DIR,
        conf_thresh=0.4  # 可根据需要调整
    )

    print("Processing completed!")
