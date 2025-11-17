from ultralytics import YOLO

models = YOLO('yolov8n.pt')
models.train(
    data=r'D:\Code\Python\Project\Multi_cam\data\data.yaml',
    epochs = 200,
    patience = 200,
    imgsz = 640,
    project = 'models',
    name = 'box_detector',
    exist_ok = True,
    mosaic = 1.0,
    mixup = 0.2,
    hsv_h = 0.015,
    hsv_s = 0.7,
    hsv_v = 0.4,
    fliplr = 0.5,
    degrees = 10,
    scale = 0.5,
    shear = 2.0,
    #device = [0],
)