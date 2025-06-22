# YOLOv12-Tomato 番茄果串检测模型

本项目为YOLOv12特化版本，针对温室番茄果串检测场景优化，利用果串空间位置关系先验知识改进检测和分类效果。

## 特性

- **相对高度回归**：额外预测番茄在果串中的相对高度位置(0~1)
- **成熟度排序约束**：利用"由上到下成熟度递减"的先验知识约束模型预测
- **特殊处理底部两颗**：强化底部果实与整串成熟度的联系
- **专用损失函数**：整合位置回归和排序约束的多任务损失函数
- **相似类别增强区分**：特殊惩罚机制增强Fully_Ripe和Ripe这两种颜色相近类别的区分能力
- **统一检测框架**：同时识别单个番茄和整个果串

## 类别定义

本模型同时支持单个番茄果实和整串番茄的检测：

### 单果类别 (4种)
- **0: Fully_Ripe** - 完全成熟番茄
- **1: Ripe** - 成熟番茄
- **2: Breaking** - 转色期番茄
- **3: Green** - 未成熟绿色番茄

### 果串类别 (2种)
- **4: Ripe_Bunch** - 成熟果串
- **5: Unripe_Bunch** - 未成熟果串

## 数据集要求

除了常规的边界框和类别标注外，还需要以下额外信息：

1. **cluster_id**：每个番茄所属的果串ID，同一图像中不同果串的ID不同
2. **h_rel**：番茄在果串中的相对高度值(0~1)，0表示最顶端，1表示最底端

### 标签格式

YOLO格式txt中每行数据为：
```
<class_id> <x_center> <y_center> <width> <height> <cluster_id> <h_rel>
```

其中：
- class_id: 类别ID (0-5)
- x_center, y_center, width, height: 归一化的边界框坐标和尺寸 (0~1)
- cluster_id: 果串ID，同一串的番茄共享同一ID
- h_rel: 相对高度值，单果类别(0-3)的范围是0~1，0表示最顶端，1表示最底端
       果串类别(4-5)应设为-1表示其相对高度无意义

例如：
```
0 0.342 0.419 0.126 0.137 1 0.15  # 果串1的顶部番茄，类别为Fully_Ripe
1 0.347 0.538 0.133 0.142 1 0.50  # 果串1的中部番茄，类别为Ripe
3 0.353 0.664 0.128 0.136 1 0.95  # 果串1的底部番茄，类别为Green
4 0.470 0.520 0.250 0.380 2 -1.0  # 果串2 (整串标注)，类别为Ripe_Bunch，h_rel为-1
```

## 训练命令

```bash
# 使用TomatoDetectWithRankLoss训练
python train.py --data tomato_dataset.yaml --cfg ultralytics/cfg/models/v12/yolov12-tomato.yaml --batch 16 --epoch 100 --loss TomatoDetectWithRankLoss

# 或指定权重参数
python train.py --data tomato_dataset.yaml --cfg ultralytics/cfg/models/v12/yolov12-tomato.yaml --batch 16 --epoch 100 --loss "TomatoDetectWithRankLoss(lambda_pos=0.3, lambda_rank=0.2, lambda_conf=0.4)"

# 参数说明:
# - lambda_pos: 控制相对高度回归损失的权重
# - lambda_rank: 控制成熟度排序约束损失的权重
# - lambda_conf: 控制Fully_Ripe和Ripe混淆惩罚的权重，增大可提高区分能力
# - margin: 成熟度排序时的边界大小
```

## 自定义数据集格式

修改您的dataset.yaml文件，确保格式如下：

```yaml
# 数据路径
path: ../datasets/tomato
train: images/train
val: images/val
test: images/test

# 类别信息
names:
  0: Fully_Ripe     # 完全成熟番茄
  1: Ripe           # 成熟番茄
  2: Breaking       # 转色期番茄
  3: Green          # 未成熟绿色番茄
  4: Ripe_Bunch     # 成熟果串
  5: Unripe_Bunch   # 未成熟果串

# 附加信息
has_cluster_id: True  # 启用果串ID
has_h_rel: True       # 启用相对高度
```

## 预处理脚本

提供了脚本用于从常规检测数据集生成cluster_id和h_rel标签：

```python
from ultralytics.utils.tomato_utils import generate_cluster_labels

# 通过计算垂直空间距离自动聚类划分果串
generate_cluster_labels(
    dataset_path="path/to/dataset",
    save_path="path/to/output", 
    cluster_method="dbscan",
    distance_threshold=0.2
)
```

## 评估指标

除了常规的mAP和召回率外，还添加了特定于番茄果串的指标：

- **串内排序正确率 (Rank Accuracy)**: 评估模型对同一果串中番茄成熟度排序的准确性
- **顶底识别准确率 (Top-Bottom Accuracy)**: 特别关注果串顶部和底部番茄识别的准确率
- **串级识别准确率 (Bunch Accuracy)**: 评估模型对整串番茄识别的准确性

## 引用

如果您在研究中使用了本项目，请引用：

```
@article{YOLOv12Tomato,
  title={YOLOv12-Tomato: Enhanced Tomato Detection with String-Level Prior Knowledge},
  author={Author},
  journal={arXiv preprint},
  year={2023}
}
``` 