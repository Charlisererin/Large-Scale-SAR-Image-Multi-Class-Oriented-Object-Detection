import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/yolo11l-obb.yaml')
    #model = YOLO('runs/train/exp/weights/last.pt')
    #如果需要恢复训练
    model.load('yolo11l-obb.pt')
    model.train(data='/home/aic/sar/dota_dataset.yaml',
                cache=False,
                imgsz=1024,
                epochs=250,
                batch=8,
                close_mosaic=0,
                workers=8,
                device='1',
                optimizer='SGD',
                patience=100,
                hsv_h=0,
                hsv_s=0,
                hsv_v=0,
                resume=True,
                amp=True,
                project='runs/train',
                name='exp',
                )
