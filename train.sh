python train.py 
--model ultralytics/cfg/models/v12/yolov12-tomato.yaml \
--data ultralytics/cfg/datasets/tomato-dataset.yaml \
--batch 16 \
--epochs 100 \
--loss "TomatoDetectWithRankLoss(lambda_pos=0.2, lambda_rank=0.3, lambda_conf=0.5)"