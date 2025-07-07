#!/usr/bin/env python3
"""
修复TomatoValidator中的target_img键缺失问题
"""

with open('ultralytics/models/yolo/tomato/val.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 查找需要修改的部分
target = """                # 更新统计信息
                # 使用append方法而不是索引赋值
                self.stats['tp'].append(correct_tensor)  # 使用torch.Tensor
                self.stats['conf'].append(predn[:, 4])
                self.stats['pred_cls'].append(predn[:, 5])
                self.stats['target_cls'].append(stat["target_cls"])"""

replacement = """                # 更新统计信息
                # 使用append方法而不是索引赋值
                self.stats['tp'].append(correct_tensor)  # 使用torch.Tensor
                self.stats['conf'].append(predn[:, 4])
                self.stats['pred_cls'].append(predn[:, 5])
                self.stats['target_cls'].append(stat["target_cls"])
                # 添加target_img键以兼容父类的get_stats方法
                if 'target_img' not in self.stats:
                    self.stats['target_img'] = []
                self.stats['target_img'].append(stat["target_img"])"""

# 替换内容
modified_content = content.replace(target, replacement)

# 写回文件
with open('ultralytics/models/yolo/tomato/val.py', 'w', encoding='utf-8') as f:
    f.write(modified_content)

print("已成功添加target_img键到self.stats字典") 