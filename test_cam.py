import cv2
import threading
import time
from ultralytics import YOLO

# ====== CẤU HÌNH ======
CAMERAS = {
    "cam1": "rtsp://admin:YDVFNP@192.168.0.108:554/ch1/main",
    # "cam2": "rtsp://admin:PBPBND@192.168.0.119:554/ch1/main",
    # "cam3": "rtsp://admin:IFPREC@192.168.0.101:554/ch1/main",
}

MODEL_PATH = r"D:\Code\Python\Project\Multi_cam\runs\detect\box_detector\weights\best.pt"
SHOW_FPS = True
DISPLAY_SCALE = 0.3   # 👈 Tỉ lệ thu nhỏ (0.4 = 40%)

# ====== NẠP MODEL ======
model = YOLO(MODEL_PATH)  # nếu không có GPU thì bỏ ".to('cuda')"

def process_camera(cam_name, cam_url):
    cap = cv2.VideoCapture(cam_url)
    if not cap.isOpened():
        print(f"❌ Không thể kết nối tới {cam_name}")
        return

    print(f"🎥 Bắt đầu detect từ {cam_name} ...")

    prev_time = time.time()
    fps = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ Mất tín hiệu từ {cam_name}")
            break

        # === DÒ ĐỐI TƯỢNG ===
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()

        # === TÍNH FPS ===
        frame_count += 1
        if frame_count >= 10:
            curr_time = time.time()
            fps = 10 / (curr_time - prev_time)
            prev_time = curr_time
            frame_count = 0

        # === HIỂN THỊ FPS ===
        if SHOW_FPS:
            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        # === THU NHỎ KÍCH THƯỚC HIỂN THỊ ===
        h, w = annotated_frame.shape[:2]
        display_frame = cv2.resize(annotated_frame, (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)))

        # Hiển thị
        cv2.imshow(cam_name, display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(cam_name)
    print(f"✅ Dừng {cam_name}")

# ====== CHẠY ĐA LUỒNG ======
threads = []
for name, url in CAMERAS.items():
    t = threading.Thread(target=process_camera, args=(name, url))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

cv2.destroyAllWindows()
