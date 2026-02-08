# Multi-Camera Object Detection System 📸🚀

Hệ thống phát hiện vật thể thời gian thực (Real-time Object Detection) hỗ trợ nhiều camera cùng lúc, sử dụng mô hình YOLO (Ultralytics) và xử lý đa luồng (Multi-threading).

## 🌟 Tính năng chính
* **Đa luồng Camera:** Hỗ trợ đọc và xử lý nhiều luồng video (RTSP/Webcam) song song nhờ `camera_thread.py`.
* **Nhận diện mạnh mẽ:** Tích hợp mô hình YOLO (v8/v11) để phát hiện vật thể với độ chính xác cao.
* **Tools hỗ trợ Training:** Bộ công cụ tích hợp sẵn trong `train_processing/` giúp tự động gán nhãn (Auto Labeling), chuyển đổi định dạng (JSON to TXT), và chia tập dữ liệu.
* **Tối ưu hóa:** Hỗ trợ chuyển đổi mô hình sang ONNX/TensorRT (`convert_onnx.py`, `convert_engine.txt`) để tăng tốc độ suy luận.

## 🛠️ Yêu cầu hệ thống
* Python 3.8 trở lên
* CUDA (Khuyến nghị nếu chạy trên GPU NVIDIA)
* Thư viện chính: `ultralytics`, `opencv-python`, `numpy`, `torch`.

## ⚙️ Cài đặt

1.  **Clone dự án về máy:**
    ```bash
    git clone [https://github.com/Spade1odin205/MultiCam-To-Detect_Object.git](https://github.com/Spade1odin205/MultiCam-To-Detect_Object.git)
    cd MultiCam-To-Detect_Object
    ```

2.  **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Data train va model đã train:**
    ```bash
    https://drive.google.com/drive/folders/1EDelp8pf3XsI6Z5Elu27efe4yawsr8vq?usp=sharing # Data dùng để train

    https://drive.google.com/drive/folders/1nyGwj-p47fYFgBLI4iohs8dsv92jLJhu?usp=sharing # Model đã train
    ```

## 🚀 Hướng dẫn sử dụng

### 1. Cấu hình Camera
Mở file `List_cam.txt` và thêm đường dẫn của các camera bạn muốn chạy (mỗi dòng một camera).


### 2. Chạy chương trình chính
Để bắt đầu nhận diện, chạy lệnh:
```bash
python final.py
3. Công cụ hỗ trợ Training (Optional)
Nếu bạn muốn train model riêng, hãy tham khảo thư mục train_processing/:

train.py: Script để bắt đầu training model YOLO.

tngoc_tools/auto_label_bb.py: Tự động gán nhãn Bounding Box.

tngoc_tools/split_train.py: Chia dữ liệu thành tập Train/Val.

📂 Cấu trúc dự án
MultiCam-To-Detect_Object/
├── data_processing/        # Xử lý luồng camera và hình ảnh
│   ├── camera_thread.py    # Class xử lý đa luồng cho camera
│   ├── camera.py           # Class camera cơ bản
│   └── ...
├── train_processing/       # Các công cụ chuẩn bị dữ liệu train
│   ├── tngoc_tools/        # Bộ tool convert, auto-label
│   ├── convert_onnx.py     # Xuất model sang ONNX
│   └── train.py            # Script training
├── List_cam.txt            # Danh sách cấu hình camera input
├── final.py                # File chạy chính của chương trình
└── requirements.txt        # Danh sách thư viện
---
