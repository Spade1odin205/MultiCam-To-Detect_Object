from ultralytics import YOLO

model = YOLO("D:\\Code\\Python\\Project\\Multi_cam\\model_new\\box_detector_6\\weights\\best.pt")

model.export(
    format="onnx",
    imgsz=640,
    opset=11,
    simplify=True,
    dynamic=False,
    batch = 1,
)
