import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('runs/train/DMFB+PIoU+FT/weights/best.pt')
    model.val(data='dataset/data.yaml',
              split='val',
              imgsz=640,
              batch=4,
              project='runs/val',
              name='DMFB+PIoU+FT',
              save_json = True
              )