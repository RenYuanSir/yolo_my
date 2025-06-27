#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import re
from pathlib import Path

# 设置日志级别
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PatchLoss")

def patch_loss_file(file_path):
    """修补loss.py文件"""
    logger.info(f"开始修补文件: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return False
    
    # 读取原文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 创建备份
    backup_path = f"{file_path}.bak"
    logger.info(f"创建备份文件: {backup_path}")
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # 修改1: 针对_prepare_targets方法中的格式检测部分
    # 查找模式: 从"检查坐标格式 - XYXY还是XYWH"开始，到"else"结尾的代码块
    pattern1 = r'(\s+# 检查坐标格式 - XYXY还是XYWH.*?)(\s+# 根据格式进行标准化.*?)(\s+if xyxy_format:.*?)(\s+else:)(.*?)(\s+# 整合归一化后的边界框\s+batch\[\'bboxes\'\] = torch\.stack\(\[cx, cy, w, h\], dim=1\))'
    
    # 替换为直接处理XYWH格式的代码
    replacement1 = r'\1\2\4\5\6'
    
    # 执行替换
    new_content = re.sub(pattern1, replacement1, content, flags=re.DOTALL)
    
    # 修改2: 针对判断是否为XYXY格式的代码块
    pattern2 = r'(\s+is_xyxy = False.*?)(\s+if xy_max > 0\.8 and wh_max > 0\.8:.*?)(\s+# 处理边界框\s+if is_xyxy:.*?)(\s+# 不管是原始XYWH)'
    
    # 替换为直接假设是XYWH格式
    replacement2 = r'\1\4'
    
    # 执行替换
    new_content = re.sub(pattern2, replacement2, new_content, flags=re.DOTALL)
    
    # 写入修改后的内容
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    logger.info(f"文件修补完成: {file_path}")
    return True

def main():
    """主函数"""
    # 获取脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建loss.py文件路径
    loss_file = os.path.join(script_dir, 'ultralytics', 'utils', 'loss.py')
    
    # 修补文件
    if patch_loss_file(loss_file):
        logger.info("修补成功!")
        return 0
    else:
        logger.error("修补失败!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 