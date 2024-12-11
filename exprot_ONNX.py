from ultralytics import RTDETR

model = RTDETR("runs/train/origin-rtdetr/weights/best.pt")
model.export(format = "onnx")