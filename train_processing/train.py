from ultralytics import YOLO

# if __name__ == '__main__':   # BẮT BUỘC TRÊN WINDOWS
#     model = YOLO('yolov8n.pt')

#     model.train(
#         data=r'D:\Code\Python\Project\Multi_cam\data_new\data\dataset_split_1\data1.yaml',
#         epochs=200,
#         patience=50,
#         imgsz=640,

#         project='model_new',
#         name='box_detector',
#         exist_ok=True,

#         # augmentation
#         hsv_h=0.005,
#         hsv_s=0.3,
#         hsv_v=0.3,
#         mosaic=0.5,
#         mixup=0.0,
#         fliplr=0.1,
#         degrees=5,
#         scale=0.2,
#         shear=0.0,

#         # device
#         device='cuda:0',

#         # Thêm để tránh lỗi multiprocessing Windows
#         workers=0,        # 🔥 quan trọng
#         batch=8           # phù hợp GTX 1650 4GB
#    )

# if __name__ == '__main__':   # BẮT BUỘC TRÊN WINDOWS
#     model = YOLO('D:\\Code\\Python\\Project\\Multi_cam\\model_new\\box_detector\\weights\\best.pt')

#     model.train(
#         data=r'D:\Code\Python\Project\Multi_cam\data_new\data\dataset_split_2\data2.yaml',
#         epochs=100,
#         patience=50,
#         imgsz=640,

#         project='model_new',
#         name='box_detector',
#         exist_ok=True,

#         # augmentation
#         hsv_h=0.005,
#         hsv_s=0.3,
#         hsv_v=0.3,
#         mosaic=0.5,
#         mixup=0.0,
#         fliplr=0.1,
#         degrees=5,
#         scale=0.2,
#         shear=0.0,

#         # device
#         device='cuda:0',

#         # Thêm để tránh lỗi multiprocessing Windows
#         workers=0,        # 🔥 quan trọng
#         batch=8           # phù hợp GTX 1650 4GB
#     )
    
# if __name__ == '__main__':   # BẮT BUỘC TRÊN WINDOWS
#     model = YOLO('D:\\Code\\Python\\Project\\Multi_cam\\model_new\\box_detector\\weights\\best.pt')

#     model.train(
#         data=r'D:\Code\Python\Project\Multi_cam\data_new\data\dataset_split_3\data3.yaml',
#         epochs=50,
#         patience=50,
#         imgsz=640,

#         project='model_new',
#         name='box_detector3',
#         exist_ok=True,

#         # augmentation
#         hsv_h=0.005,
#         hsv_s=0.3,
#         hsv_v=0.3,
#         mosaic=0.5,
#         mixup=0.0,
#         fliplr=0.1,
#         degrees=5,
#         scale=0.2,
#         shear=0.0,

#         # device
#         device='cuda:0',

#         # Thêm để tránh lỗi multiprocessing Windows
#         workers=0,        # 🔥 quan trọng
#         batch=8           # phù hợp GTX 1650 4GB
#     )

# if __name__ == '__main__':   # BẮT BUỘC TRÊN WINDOWS
#     model = YOLO('D:\\Code\\Python\\Project\\Multi_cam\\model_new\\box_detector\\weights\\best.pt')

#     model.train(
#         data=r'D:\Code\Python\Project\Multi_cam\data_new\data\dataset_split_4\data4.yaml',
#         epochs=20,
#         patience=20,
#         imgsz=640,

#         project='model_new',
#         name='box_detector4',
#         exist_ok=True,

#         # augmentation
#         hsv_h=0.005,
#         hsv_s=0.3,
#         hsv_v=0.3,
#         mosaic=0.5,
#         mixup=0.0,
#         fliplr=0.1,
#         degrees=5,
#         scale=0.2,
#         shear=0.0,

#         # device
#         device='cuda:0',

#         # Thêm để tránh lỗi multiprocessing Windows
#         workers=0,        # 🔥 quan trọng
#         batch=8           # phù hợp GTX 1650 4GB
#     )

# if __name__ == '__main__':   # BẮT BUỘC TRÊN WINDOWS
#     model = YOLO('D:\\Code\\Python\\Project\\Multi_cam\\model_new\\box_detector3\\weights\\best.pt')

#     model.train(
#         data=r'D:\Code\Python\Project\Multi_cam\data_new\data\dataset_split_5\data5.yaml',
#         epochs=50,
#         patience=50,
#         imgsz=640,

#         project='model_new',
#         name='box_detector5',
#         exist_ok=True,

#         # augmentation
#         hsv_h=0.005,
#         hsv_s=0.3,
#         hsv_v=0.3,
#         mosaic=0.5,
#         mixup=0.0,
#         fliplr=0.1,
#         degrees=5,
#         scale=0.2,
#         shear=0.0,

#         # device
#         device='cuda:0',

#         # Thêm để tránh lỗi multiprocessing Windows
#         workers=0,        # 🔥 quan trọng
#         batch=8           # phù hợp GTX 1650 4GB
#     )

from ultralytics import YOLO

if __name__ == '__main__':
    # Load model gốc để train sạch từ đầu
    model = YOLO('yolov8n.pt')

    model.train(
        # Nhớ dùng file yaml chứa TẤT CẢ dữ liệu (Gộp 5 phần lại)
        data=r'D:\Code\Python\Project\Multi_cam\data_new\data\dataset_split_5\data5.yaml',
        
        epochs=150,        
        patience=50,
        imgsz=640,

        project='model_new',
        name='box_detector_6',
        exist_ok=True,

        # --- CHIẾN LƯỢC MOSAIC (QUAN TRỌNG NHẤT) ---
        mosaic=1.0,         # Ép học 100% dạng lưới (giả lập 4 camera)
        close_mosaic=20,    # 20 epoch cuối tắt mosaic để tinh chỉnh ảnh thật

        # --- AUGMENTATION HÌNH HỌC ---
        scale=0.5,          # (Tăng lên) Giúp nhận diện vật thể nhỏ/to linh hoạt
        degrees=10.0,       # Xoay nhẹ +/- 10 độ
        fliplr=0.5,         # Lật ngang ảnh (tăng gấp đôi dữ liệu)
        flipud=0.0,         # Lật dọc (thường không cần trừ khi hộp lộn ngược)
        shear=0.0,          # Không làm méo ảnh
        
        # --- AUGMENTATION MÀU SẮC & NHIỄU ---
        hsv_h=0.015,        # Thay đổi tông màu nhẹ (tránh đổi màu hộp quá đà)
        hsv_s=0.4,          # Thay đổi độ đậm nhạt màu
        hsv_v=0.4,          # Thay đổi độ sáng (quan trọng cho camera thực tế)
        mixup=0.1,          # Trộn ảnh (giúp chống nhiễu nhẹ)
        
        # --- CẤU HÌNH HỆ THỐNG ---
        device='cuda:0',
        workers=0,
        batch=8 # Hoặc 16
    )