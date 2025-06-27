#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
from pathlib import Path

# 设置日志级别
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FixTomatoModel")

def fix_tomato_model():
    """修改TomatoDetectionModel.forward方法，使其始终能接受augment参数"""
    try:
        # 导入模块
        from ultralytics.nn.tasks import TomatoDetectionModel
        
        # 获取原始forward方法
        original_forward = TomatoDetectionModel.forward
        
        # 定义新的forward方法
        def new_forward(self, x, augment=False, profile=False, visualize=False, embed=None):
            """
            确保forward方法支持augment参数，兼容标准YOLO API
            
            Args:
                x: 输入张量
                augment: 是否启用增强推理
                profile: 是否启用性能分析
                visualize: 是否可视化特征图
                embed: 是否返回特征嵌入
                
            Returns:
                处理结果
            """
            if augment:
                return self._predict_augment(x)
            
            # 调用原始forward方法，但只传入它支持的参数
            return original_forward(self, x)
        
        # 替换方法
        TomatoDetectionModel.forward = new_forward
        
        logger.info("成功修复TomatoDetectionModel.forward方法")
        return True
    except Exception as e:
        logger.error(f"修复TomatoDetectionModel失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def fix_loss_prepare_targets():
    """修复TomatoDetectWithRankLoss._prepare_targets方法，确保始终按XYWH格式处理"""
    try:
        # 找到loss.py文件
        script_dir = os.path.dirname(os.path.abspath(__file__))
        loss_file = os.path.join(script_dir, 'ultralytics', 'utils', 'loss.py')
        
        if not os.path.exists(loss_file):
            logger.error(f"找不到loss.py文件: {loss_file}")
            return False
        
        # 读取文件内容
        with open(loss_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 创建备份
        backup_file = loss_file + '.bak'
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"已创建备份文件: {backup_file}")
        
        # 在prepare_targets方法中查找并替换格式检测和转换的代码
        
        # 修改1: 移除XYXY相关的判断
        content = content.replace(
            'is_xyxy = False\n                if xy_max > 0.8 and wh_max > 0.8:',
            'is_xyxy = False  # 所有标签都按XYWH格式处理\n                if False:  # 禁用XYXY检测'
        )
        
        # 修改2: 移除XYXY转换代码
        content = content.replace(
            '# 处理边界框\n                if is_xyxy:',
            '# 处理边界框\n                if False:  # 禁用XYXY转换'
        )
        
        # 写入修改后的内容
        with open(loss_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"成功修复{loss_file}中的格式检测和转换代码")
        return True
    except Exception as e:
        logger.error(f"修复_prepare_targets方法失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """主函数"""
    success = True
    
    # 修复TomatoDetectionModel.forward方法
    if not fix_tomato_model():
        success = False
    
    # 修复_prepare_targets方法
    if not fix_loss_prepare_targets():
        success = False
    
    if success:
        logger.info("所有修复已成功应用")
        return 0
    else:
        logger.error("部分修复失败")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 