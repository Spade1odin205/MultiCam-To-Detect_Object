import cv2
import threading
import time
import numpy as np

# ================= CẤU HÌNH CAMERA =================
# Thay đổi danh sách này theo IP thực tế của bạn
CAMERAS = [
    "rtsp://admin:EIUSAY@192.168.1.143:554/ch1/main",
    "rtsp://admin:DTAJVP@192.168.1.124:554/ch1/main",
    "rtsp://admin:NXKPHU@192.168.1.157:554/ch1/main",
    "rtsp://admin:HLTHKD@192.168.1.127:554/ch1/main",
    # Thêm camera khác nếu muốn test 8 cam
]

# Kích thước ảnh đầu ra mong muốn (Nên để nhỏ để nhẹ tải hiển thị)
DISPLAY_W, DISPLAY_H = 640, 360

class CameraThread(threading.Thread):
    def __init__(self, url, index):
        super().__init__()
        self.url = url
        self.index = index
        self.frame = None
        self.running = True
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # --- PIPELINE GSTREAMER TỐI ƯU CHO JETSON NANO ---
        # 1. nvv4l2decoder: Giải mã bằng GPU
        # 2. nvvidconv: Resize bằng GPU (quan trọng nhất để tăng FPS)
        # 3. video/x-raw: Chuyển về kích thước nhỏ ngay tại đây
        self.gst_pipeline = (
            f"rtspsrc location={self.url} latency=200 ! "
            f"rtph265depay ! h265parse ! nvv4l2decoder ! "
            f"nvvidconv ! video/x-raw,format=BGRx,width={DISPLAY_W},height={DISPLAY_H} ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"appsink drop=true sync=false max-buffers=1"
        )

    def run(self):
        print(f"[Cam {self.index}] Connecting...")
        cap = cv2.VideoCapture(self.gst_pipeline, cv2.CAP_GSTREAMER)
        
        if not cap.isOpened():
            print(f"[Cam {self.index}] ❌ Failed to open!")
            return

        while self.running:
            ret, frame = cap.read()
            if ret:
                self.frame = frame
                
                # Tính FPS riêng cho từng camera
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - self.start_time
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.start_time = time.time()
            else:
                # Nếu mất kết nối, chờ nhẹ rồi thử lại (tránh treo vòng lặp)
                time.sleep(0.1)

        cap.release()
        print(f"[Cam {self.index}] Stopped.")

    def get_frame(self):
        return self.frame, self.fps

    def stop(self):
        self.running = False
        self.join()

def main():
    threads = []
    
    # Khởi động các luồng camera
    print(f"🚀 Starting {len(CAMERAS)} streams with Hardware Resizing...")
    for i, url in enumerate(CAMERAS):
        t = CameraThread(url, i)
        t.start()
        threads.append(t)

    # Đợi cam khởi động ổn định
    time.sleep(3)
    
    # Tạo ảnh đen chờ sẵn (Placeholder)
    blank_image = np.zeros((DISPLAY_H, DISPLAY_W, 3), np.uint8)

    try:
        while True:
            loop_start = time.time()
            
            frames_list = []
            total_fps = 0
            
            # Lấy frame từ các luồng
            for t in threads:
                frame, fps = t.get_frame()
                total_fps += fps
                
                if frame is None:
                    # Nếu chưa có frame thì hiện màn hình đen + text
                    img = blank_image.copy()
                    cv2.putText(img, "No Signal", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    img = frame
                    # Vẽ FPS lên góc ảnh
                    cv2.putText(img, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                frames_list.append(img)

            # Ghép ảnh để hiển thị (Grid Layout đơn giản)
            # Tự động tính toán chia dòng (mỗi dòng tối đa 2 hoặc 3 ảnh)
            cols = 2
            rows = (len(frames_list) + cols - 1) // cols
            
            # Bổ sung ảnh đen nếu thiếu ô cho đủ lưới
            while len(frames_list) < rows * cols:
                frames_list.append(blank_image)

            # Ghép ma trận ảnh (Numpy black magic - siêu nhanh)
            grid_rows = []
            for r in range(rows):
                row_imgs = frames_list[r*cols : (r+1)*cols]
                grid_rows.append(np.hstack(row_imgs))
            
            final_view = np.vstack(grid_rows)

            # Hiển thị
            cv2.imshow("Jetson Nano Multi-Stream Test", final_view)

            # Tính toán độ trễ hiển thị tổng
            process_time = (time.time() - loop_start) * 1000
            # print(f"Display Latency: {process_time:.1f} ms") # Bật dòng này nếu muốn xem độ trễ

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping all threads...")
        for t in threads:
            t.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()