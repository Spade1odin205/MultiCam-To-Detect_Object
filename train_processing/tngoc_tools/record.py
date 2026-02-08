import cv2

# ====== RTSP CAMERA ======
rtsp_url = "rtsp://admin:TIJEQB@192.168.1.134:554/ch1/main"

cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("❌ Không mở được luồng RTSP.")
    exit()

# ====== LẤY KÍCH THƯỚC GỐC ======
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0 or fps > 120:
    fps = 30

print(f"Camera size: {w}x{h}, FPS: {fps}")

# ====== SETUP VIDEO WRITER ======
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_video_new126543.mp4", fourcc, fps, (w, h))

if not out.isOpened():
    print("❌ Không mở được VideoWriter.")
    exit()

# ====== LOOP GHI VIDEO ======
print("🎥 Bắt đầu ghi video... Nhấn Q để dừng.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠ Mất frame từ camera.")
        break

    # --- GHI VIDEO GỐC ---
    out.write(frame)

    # --- HIỂN THỊ 1280x720 ---
    frame_display = cv2.resize(frame, (1280, 720))
    cv2.imshow("RTSP Stream (1280x720)", frame_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ====== GIẢI PHÓNG ======
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Đã ghi xong video!")
