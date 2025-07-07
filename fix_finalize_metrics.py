#!/usr/bin/env python3
"""
修复TomatoValidator中的finalize_metrics方法
"""

with open('ultralytics/models/yolo/tomato/val.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 查找需要修改的部分
target = """    def finalize_metrics(self, *args, **kwargs):
        \"\"\"完成指标计算，添加h_pos评估和排序指标。\"\"\"
        # 调用父类方法处理常规检测指标
        metrics = super().finalize_metrics(*args, **kwargs)
        
        # 使用TomatoMetrics的finalize_h_pos_metrics方法
        self.metrics.finalize_h_pos_metrics()
        
        # 添加h_pos指标到结果中
        h_mae = self.metrics.h_pos_stats.get('h_mae', 0.0)
        rank_acc = self.metrics.h_pos_stats.get('rank_acc', 0.0)
        
        metrics['metrics/h_mae'] = h_mae
        metrics['metrics/rank_acc'] = rank_acc
        
        LOGGER.info(f"H-pos MAE: {h_mae:.4f}")
        LOGGER.info(f"H-pos Ranking Accuracy: {rank_acc:.4f}")
        
        return metrics"""

replacement = """    def finalize_metrics(self, *args, **kwargs):
        \"\"\"完成指标计算，添加h_pos评估和排序指标。\"\"\"
        # 调用父类方法处理常规检测指标
        super().finalize_metrics(*args, **kwargs)
        
        # 使用TomatoMetrics的finalize_h_pos_metrics方法
        self.metrics.finalize_h_pos_metrics()
        
        # 添加h_pos指标到结果中
        h_mae = self.metrics.h_pos_stats.get('h_mae', 0.0)
        rank_acc = self.metrics.h_pos_stats.get('rank_acc', 0.0)
        
        # 创建一个新的字典来存储结果，而不是尝试修改父类方法的返回值
        metrics = {}
        metrics['metrics/h_mae'] = h_mae
        metrics['metrics/rank_acc'] = rank_acc
        
        LOGGER.info(f"H-pos MAE: {h_mae:.4f}")
        LOGGER.info(f"H-pos Ranking Accuracy: {rank_acc:.4f}")
        
        return metrics"""

# 替换内容
modified_content = content.replace(target, replacement)

# 写回文件
with open('ultralytics/models/yolo/tomato/val.py', 'w', encoding='utf-8') as f:
    f.write(modified_content)

print("已成功修复finalize_metrics方法") 