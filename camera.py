import cv2
import threading
import time
import numpy as np
from collections import deque

# ==================== CẤU HÌNH TỐI ƯU CHO JETSON NANO ====================
DISPLAY_WIDTH = 320   
DISPLAY_HEIGHT = 240 
RESIZE_DIM = (DISPLAY_WIDTH, DISPLAY_HEIGHT)

# Buffer size - quan trọng cho Jetson Nano (RAM hạn chế)
FRAME_BUFFER_SIZE = 2  # Chỉ giữ 2 frames gần nhất

RTSP_URLS = [
    "rtsp://admin:NNFVAJ@192.168.0.114:ch1/main",
    "rtsp://admin:CYXJBA@192.168.0.109:ch1/main",
    "rtsp://admin:CPSFLT@192.168.0.104:ch1/main",
    "rtsp://admin:WSLRQC@192.168.0.113:ch1/main",
]

class CameraStream:
    def __init__(self, rtsp_url, stream_id, resize_dim):
        self.rtsp_url = rtsp_url
        self.stream_id = stream_id
        self.resize_dim = resize_dim
        self.cap = None
        self.frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)
        self.running = False
        self.thread = None
        self.frame_count = 0
        self.start_time = time.time()
        self.capture_fps = 0.0
        self.reconnect_attempts = 0
        self.max_reconnect = 3

    def create_optimized_pipeline(self, rtsp_url, resize_dim):
        """
        Pipeline tối ưu cho Jetson Nano + H.265
        Key optimizations:
        1. Zero-copy với NVMM memory
        2. Hardware decode (nvv4l2decoder)
        3. Hardware scaling (nvvidconv)
        4. Output BGR trực tiếp (tương thích với code cũ)
        """
        width, height = resize_dim
        
        pipeline = (
            f"rtspsrc location={rtsp_url} "
            "latency=200 "  # Giữ 200ms cho ổn định
            "protocols=tcp ! "
            
            # H.265 decode
            "rtph265depay ! "
            "h265parse ! "
            "nvv4l2decoder enable-max-performance=1 ! "
            
            # Hardware scaling trong NVMM memory
            "nvvidconv ! "
            f"video/x-raw(memory:NVMM), width={width}, height={height} ! "
            
            # Convert sang BGR (system memory) - OpenCV cần BGR
            "nvvidconv ! "
            "video/x-raw, format=BGRx ! "  # BGRx thay vì I420
            
            "videoconvert ! "  # Final convert
            "video/x-raw, format=BGR ! "
            
            "appsink "
            "max-buffers=2 "
            "drop=true "
            "sync=false"
        )
        
        return pipeline

    def open(self):
        pipeline = self.create_optimized_pipeline(self.rtsp_url, self.resize_dim)
        print(f"🚀 Đang mở camera {self.stream_id}...")
        
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        
        if self.cap.isOpened():
            # Tối ưu buffer cho VideoCapture
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"✅ Camera {self.stream_id} online | Pipeline: H.265→I420")
            self.running = True
            self.thread = threading.Thread(target=self.update, daemon=True)
            self.thread.start()
            return True
        else:
            print(f"❌ Camera {self.stream_id} failed ({self.rtsp_url})")
            return False

    def update(self):
        """Thread loop - đọc frames liên tục"""
        consecutive_fails = 0
        
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                consecutive_fails += 1
                if consecutive_fails > 30:  # 30 frames liên tiếp fail
                    print(f"⚠️ Camera {self.stream_id} mất kết nối, đang reconnect...")
                    self.reconnect()
                    consecutive_fails = 0
                time.sleep(0.01)
                continue
            
            consecutive_fails = 0
            
            # Thêm frame vào buffer (tự động drop frame cũ nhất)
            self.frame_buffer.append(frame)
            
            # Tính FPS
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            if elapsed >= 1.0:
                self.capture_fps = self.frame_count / elapsed
                self.frame_count = 0
                self.start_time = time.time()

    def read(self):
        """Lấy frame mới nhất từ buffer"""
        if len(self.frame_buffer) > 0:
            return self.frame_buffer[-1]  # Frame mới nhất
        return None

    def get_fps(self):
        return self.capture_fps

    def reconnect(self):
        """Thử reconnect camera"""
        if self.reconnect_attempts >= self.max_reconnect:
            print(f"🔴 Camera {self.stream_id} đã thử reconnect {self.max_reconnect} lần, bỏ qua.")
            return
        
        self.reconnect_attempts += 1
        if self.cap:
            self.cap.release()
        time.sleep(2)
        self.open()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        print(f"🛑 Camera {self.stream_id} stopped")


def create_grid(frames, size, cols):
    """Tạo grid hiển thị - tối ưu cho Jetson Nano"""
    w, h = size
    
    # Tạo blank frame (NO SIGNAL) - BGR format
    blank = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(blank, "NO SIGNAL", (w // 2 - 60, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    processed = []
    for f in frames:
        if f is not None and f.size > 0:
            try:
                # Frame đã là BGR từ pipeline
                processed.append(f)
            except Exception as e:
                print(f"⚠️ Error processing frame: {e}")
                processed.append(blank)
        else:
            processed.append(blank)

    # Tạo grid
    total = len(processed)
    rows = int(np.ceil(total / cols))
    for _ in range(rows * cols - total):
        processed.append(blank)
    
    grid_rows = []
    for r in range(rows):
        grid_rows.append(np.hstack(processed[r * cols:(r + 1) * cols]))
    
    return np.vstack(grid_rows)


def main():
    print("=" * 60)
    print("🚀 OPTIMIZED RTSP VIEWER FOR JETSON NANO")
    print("=" * 60)
    print(f"📹 Cameras: {len(RTSP_URLS)}")
    print(f"📐 Resolution: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT} (I420 format)")
    print(f"🎯 Target: 15+ FPS với model inference")
    print("=" * 60)
    
    num = len(RTSP_URLS)
    cols = int(np.ceil(np.sqrt(num)))

    # Khởi tạo cameras
    cameras = [CameraStream(url, i + 1, RESIZE_DIM) for i, url in enumerate(RTSP_URLS)]
    
    active_cameras = 0
    for cam in cameras:
        if cam.open():
            active_cameras += 1
    
    print(f"\n✅ {active_cameras}/{num} cameras active")
    
    if active_cameras == 0:
        print("❌ Không có camera nào hoạt động. Thoát.")
        return

    time.sleep(2)  # Đợi cameras ổn định
    print("\n▶️  Nhấn 'q' để thoát | 'r' để reset FPS counter")

    frame_counter = 0
    start_time = time.time()
    disp_fps = 0.0

    while True:
        loop_start = time.time()
        
        # Lấy frames từ tất cả cameras
        frames = [cam.read() for cam in cameras]
        
        # Tạo mosaic
        mosaic = create_grid(frames, RESIZE_DIM, cols)

        # Tính Display FPS
        frame_counter += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            disp_fps = frame_counter / elapsed
            frame_counter = 0
            start_time = time.time()

        # Vẽ FPS info
        cv2.putText(mosaic, f"Display: {disp_fps:.1f} FPS", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Vẽ FPS từng camera
        for i, cam in enumerate(cameras):
            fps = cam.get_fps()
            row, col = divmod(i, cols)
            x = col * DISPLAY_WIDTH + 10
            y = row * DISPLAY_HEIGHT + 50
            color = (0, 255, 0) if fps > 10 else (0, 165, 255)  # Xanh nếu >10fps
            cv2.putText(mosaic, f"C{i+1}: {fps:.1f}", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow("RTSP Multi-Camera [Optimized]", mosaic)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            frame_counter = 0
            start_time = time.time()
            print("🔄 FPS counter reset")
        
        # Giới hạn loop rate (tránh quá tải CPU)
        loop_time = time.time() - loop_start
        if loop_time < 0.033:  # ~30 FPS max
            time.sleep(0.033 - loop_time)

    # Cleanup
    print("\n🛑 Đang dừng cameras...")
    for cam in cameras:
        cam.stop()
    cv2.destroyAllWindows()
    print("✅ Đã thoát sạch sẽ")


if __name__ == "__main__":
    main()
