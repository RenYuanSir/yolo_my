from ultralytics import YOLO


if __name__ == '__main__':


    # Load a model
    model = YOLO(model=r'D:\yoloProject\yolov12-turbo\runs\train\exp_flower_v1\detect_flower_v1.pt')
    model.predict(source=r'https://cdn.wanx.aliyuncs.com/wanx/1743153309171209530/text_to_image_v2/5c80226d283e4f4a9dd5e90a7c776d9e_0.png',
                  save=True,
                  show=False,
                  )
