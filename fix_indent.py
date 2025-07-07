#!/usr/bin/env python3
"""
修复val.py中的缩进错误
"""

import re

# 读取文件内容
with open('ultralytics/models/yolo/tomato/val.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 使用正则表达式查找并替换缩进错误的部分
pattern = r'(\s+)# 添加h_pos指标到结果中\n(\s+)h_mae = self\.metrics\.h_pos_stats\.get\(\'h_mae\', 0\.0\)\n(\s+)rank_acc = self\.metrics\.h_pos_stats\.get\(\'rank_acc\', 0\.0\)\n\s+\n\s+metrics\[\'metrics/h_mae\'\] = h_mae\n\s+metrics\[\'metrics/rank_acc\'\] = rank_acc\n\s+\n\s+LOGGER\.info\(f"H-pos MAE: \{h_mae:\.\4f\}"\)\n\s+LOGGER\.info\(f"H-pos Ranking Accuracy: \{rank_acc:\.\4f\}"\)'

replacement = r'        # 添加h_pos指标到结果中\n        h_mae = self.metrics.h_pos_stats.get(\'h_mae\', 0.0)\n        rank_acc = self.metrics.h_pos_stats.get(\'rank_acc\', 0.0)\n        \n        metrics[\'metrics/h_mae\'] = h_mae\n        metrics[\'metrics/rank_acc\'] = rank_acc\n        \n        LOGGER.info(f"H-pos MAE: {h_mae:.4f}")\n        LOGGER.info(f"H-pos Ranking Accuracy: {rank_acc:.4f}")'

# 直接替换有问题的部分
fixed_content = content.replace("""        # 添加h_pos指标到结果中
        h_mae = self.metrics.h_pos_stats.get('h_mae', 0.0)
        rank_acc = self.metrics.h_pos_stats.get('rank_acc', 0.0)
        
                metrics['metrics/h_mae'] = h_mae
        metrics['metrics/rank_acc'] = rank_acc
        
                LOGGER.info(f"H-pos MAE: {h_mae:.4f}")
                        LOGGER.info(f"H-pos Ranking Accuracy: {rank_acc:.4f}")""", 
"""        # 添加h_pos指标到结果中
        h_mae = self.metrics.h_pos_stats.get('h_mae', 0.0)
        rank_acc = self.metrics.h_pos_stats.get('rank_acc', 0.0)
        
        metrics['metrics/h_mae'] = h_mae
        metrics['metrics/rank_acc'] = rank_acc
        
        LOGGER.info(f"H-pos MAE: {h_mae:.4f}")
        LOGGER.info(f"H-pos Ranking Accuracy: {rank_acc:.4f}")""")

# 写入修复后的内容
with open('ultralytics/models/yolo/tomato/val.py', 'w', encoding='utf-8') as f:
    f.write(fixed_content)

print("缩进错误已修复") 