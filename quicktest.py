from ultralytics.data.dataset import YOLODataset
dset = YOLODataset(img_path=[], data='/root/shared-nvme/tomato_transStyle/data.yaml')
print(dset.has_h_rel, dset.has_cluster_id)   # 期望 True True，实际多半 False False
