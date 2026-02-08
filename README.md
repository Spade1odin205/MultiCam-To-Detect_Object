# Multi-Camera Object Detection (TensorRT Mosaic)

Hệ thống phát hiện vật thể thời gian thực cho nhiều camera, chạy đa luồng và gộp tối đa 4 camera thành một ảnh mosaic để suy luận bằng TensorRT.

## Tính năng

- Đọc nhiều luồng RTSP/Webcam song song (multi-thread).
- Gộp (mosaic) 2x2 để giảm số lần inference khi chạy nhiều camera.
- Suy luận bằng TensorRT + PyCUDA để tối ưu tốc độ.
- Có các script xử lý dữ liệu/train trong `train_processing/`.

## Yêu cầu

- Python 3.8+ (khuyến nghị 3.10/3.11 nếu phù hợp môi trường).
- NVIDIA GPU + CUDA.
- TensorRT + PyCUDA.
- OpenCV.

Lưu ý: File `requirements.txt` trong repo khá “nặng” (nhiều package không liên quan runtime). Nếu bạn chỉ muốn chạy `final.py`, bạn có thể cài tối thiểu theo đúng môi trường TensorRT của máy.

## Cài đặt

### 1) Tạo môi trường Python

Ví dụ (Windows PowerShell):

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Nếu bạn đang dùng môi trường có sẵn trong repo (ví dụ `yolo/`), hãy kích hoạt đúng environment trước khi chạy.

### 2) Data / model tham khảo

- Data dùng để train: https://drive.google.com/drive/folders/1EDelp8pf3XsI6Z5Elu27efe4yawsr8vq?usp=sharing
- Model đã train: https://drive.google.com/drive/folders/1nyGwj-p47fYFgBLI4iohs8dsv92jLJhu?usp=sharing

## Chạy chương trình

Hiện tại `final.py` KHÔNG đọc `List_cam.txt`. Danh sách camera và đường dẫn TensorRT engine đang được cấu hình trực tiếp trong code.

### 1) Cấu hình camera

Mở `final.py` và sửa biến `CAM_SOURCES`:

```python
CAM_SOURCES = [
    "rtsp://user:pass@ip:554/ch1/main",
    0,  # webcam (tuỳ máy)
]
```

### 2) Cấu hình TensorRT engine

Trong `final.py`, sửa `ENGINE_PATH` trỏ tới file `.engine` trên máy bạn.

Ghi chú quan trọng:

- `ENGINE_PATH` trong repo đang để ví dụ kiểu Linux (`/home/...`). Bạn bắt buộc phải đổi sang đường dẫn thật trên máy.
- Model input mặc định là `640x640` (xem `MODEL_W`, `MODEL_H`). Engine cần khớp kích thước này.

### 3) Chạy

```bash
python final.py
```

## Lưu ý về RTSP / GStreamer

Trong `final.py`, khi source bắt đầu bằng `rtsp`, code đang dùng pipeline GStreamer với `nvv4l2decoder` (thường dùng trên NVIDIA Jetson / môi trường có plugin tương ứng).

- Nếu bạn chạy trên Windows hoặc máy không có `nvv4l2decoder`, phần mở RTSP có thể fail.
- Cách đơn giản là chỉnh `ThreadedCamera.__init__` để dùng `cv2.VideoCapture(source)` cho RTSP (không dùng pipeline), hoặc cài đúng GStreamer + plugins tương ứng.

## Training / Tools (tuỳ chọn)

Thư mục `train_processing/` chứa các công cụ:

- `train_processing/train.py`: script train.
- `train_processing/convert_onnx.py`: export ONNX.
- `train_processing/tngoc_tools/`: auto-label, convert định dạng, chia train/val.

## Cấu trúc thư mục (tóm tắt)

```text
Multi_cam/
├── final.py
├── List_cam.txt
├── requirements.txt
├── data_processing/
├── train_processing/
└── model_new/
```
