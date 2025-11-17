import cv2

# === CẤU HÌNH ===
# Ví dụ URL: "rtsp://username:password@192.168.1.100:554/Streaming/Channels/101"
# hoặc: "http://192.168.1.100:8080/video" (với IP Webcam app)
ip_camera_url = "rtsp://admin:WXEXRW@192.168.0.111:554/ch1/main"

# === KẾT NỐI ===
cap = cv2.VideoCapture(ip_camera_url)

if not cap.isOpened():
    print("Không thể kết nối tới camera IP. Kiểm tra lại URL hoặc kết nối mạng.")
    exit()

print("Kết nối thành công. Nhấn 'q' để thoát.")

# === VÒNG LẶP HIỂN THỊ ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("Không nhận được khung hình (mất kết nối?)")
        break

    cv2.imshow("IP Camera", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === GIẢI PHÓNG TÀI NGUYÊN ===
cap.release()
cv2.destroyAllWindows()
