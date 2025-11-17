import cv2
import os

# ====== CẤU HÌNH ======
VIDEO_PATH = r"D:\Code\Python\Project\Multi_cam\video_HA\video4.mp4"  # đường dẫn video
OUTPUT_DIR = r"D:\Code\Python\Project\Multi_cam\frames2\frame_cam4"     # thư mục lưu ảnh
FRAME_INTERVAL = 5  # Mỗi 5 frame lưu 1 ảnh

# ====== TẠO THƯ MỤC LƯU ẢNH ======
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====== MỞ VIDEO ======
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Không thể mở video.")
    exit()

frame_count = 0
saved_count = 0

print("🎥 Bắt đầu cắt ảnh...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Nếu là frame thứ n thì lưu lại
    if frame_count % FRAME_INTERVAL == 0:
        filename = os.path.join(OUTPUT_DIR, f"3_4frame_{saved_count:05d}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"✅ Đã lưu {saved_count} ảnh tại: {OUTPUT_DIR}")
