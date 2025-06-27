try:
    from ultralytics.data.dataset import TomatoYOLODataset
    print('TomatoYOLODataset类已成功导入')
except Exception as e:
    print(f'导入错误: {e}') 