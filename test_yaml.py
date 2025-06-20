import warnings
warnings.filterwarnings('ignore')
import os, tqdm
from ultralytics import YOLO

if __name__ == '__main__':
    error_result = []
    for yaml_path in tqdm.tqdm(os.listdir(r'D:\yoloProject\yolov12-turbo\ultralytics\cfg\models\v12')):
        if 'rtdetr' not in yaml_path and 'cls' not in yaml_path and 'world' not in yaml_path:
            try:
                model = YOLO(f'D:/yoloProject/yolov12-turbo/ultralytics/cfg/models/v12/{yaml_path}')
                model.info(detailed=False)
                model.profile([640, 640])
                model.fuse()
            except Exception as e:
                error_result.append(f'{yaml_path} {e}')
    
    for i in error_result:
        print(i)