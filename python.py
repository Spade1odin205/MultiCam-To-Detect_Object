import cv2
import threading
import queue
import time 
import sys
import numpy as np

# ========================================================================
# <<<--- PHAN CAI DAT (1/2)     --->>>
# ========================================================================
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 360
TARGET_FPS = 15

# ========================================================================
# <<<--- PHAN CAI DAT (2/2)     --->>>
# ========================================================================

# QUAN TRONG: NEN DOI 'H265' THANH 'H264' O DAY VA DOI CAI DAT TREN CAMERA TUONG UNG

CAM_SPECS = {
    "Cam 1": {
        "url": "rtsp://admin:YDVFNP@192.168.0:554/ch1/main",
        "codec": "H265", # nen doi thanh "h264"
    },
    "Cam 2": {
        "url": "rtsp://admin:YDVFNP@192.168.0:554/ch1/main",
        "codec": "H265", # nen doi thanh "h264"
    },
    "Cam 3": {
        "url": "rtsp://admin:YDVFNP@192.168.0:554/ch1/main",
        "codec": "H265", # nen doi thanh "h264"
    },
    "Cam 4": {
        "url": "rtsp://admin:YDVFNP@192.168.0:554/ch1/main",
        "codec": "H265", # nen doi thanh "h264"
    },
}

# ========================================================================
# Ham tao gstreamer pipeline
# ========================================================================

def create_gstreamer_pipeline(specs, display_width, display_height, fps):
    url = specs["url"]
    codec = specs["codec"].lower()

    if codec == "h264":
        decoder_pipeline = "rtph264depay ! h264parse ! nvv4l2decoder"
    elif codec == "h265":
        decoder_pipeline = "rtph265depay ! h265parse ! nvv4l2decoder"
    else:
        raise ValueError(f"Unsupported codec: {codec}")

    pipeline_str = (
        f"rtspsrc location={url} latency=0 ! "
        f"{decoder_pipeline} ! "
        f"nvvidconv ! video/x-raw, width={display_width}, height={display_height}, framerate=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
    )

    return pipeline_str

# ========================================================================
# Ham doc camera tren luong rieng
# ========================================================================
def read_camera_stream(pipeline, frame_queue, stream_name):
    print(f"bat dau luong cho: {stream_name}")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print(f"Khong the mo camera: {stream_name}")
        return
    
    # --- bien tinh fps cho luong rieng nay ---
    fps_frame_count = 0
    fps_start_time = time.time()
    stream_fps = 0.0
    #-----------------------------------------
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print(f"Khong the doc frame tu camera: {stream_name}")
            cap.release()
            time.sleep(5)
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                continue
            else:
                print(f"Da ket noi lai duoc camera: {stream_name}")
                continue
            
        # Tinh fps cho luong nay
        fps_frame_count += 1
        current_time = time.time()
        elapsed_fps_time = current_time - fps_start_time
        
        if elapsed_fps_time >= 1.0:
            stream_fps = fps_frame_count / elapsed_fps_time
            fps_frame_count = 0
            fps_start_time = current_time
        # ------------------------------
        
        if frame_queue.full():
            frame_queue.get_nowait()
            
        frame_queue.put((frame, stream_fps))
        
# ========================================================================
# Chuong trinh chinh
# ========================================================================

def main():
    # 1. Tao cac pipeline
    pipelines = {
        name: create_gstreamer_pipeline(specs, DISPLAY_WIDTH, DISPLAY_HEIGHT, TARGET_FPS)
        for name, specs in CAM_SPECS.items()
    }
    
    # 2. Tao queue va khoi tao cac luong
    frame_queues = {name: queue.Queue(maxsize=1) for name in pipelines}
    threads = []
    
    for name, pipeline_str in pipelines.items():
        thread = threading.Thread(
            target=read_camera_stream,
            args=(pipeline_str, frame_queues[name], name)
            daemon=True
        )
        thread.append(thread)
        thread.start()
        
    print("Da khoi tao cac luong")
    time.sleep(5)
    
# 3. Vong lap chinh

display_fps_frame_count = 0
display_fps_start_time = time.time()
display_fps = 0.0

placeholder_frame = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
placeholder = (placeholder_frame, 0.0)

loop_delay = 1.0 / TARGET_FPS

while True:
    loop_start_time = time.time()
    
    frames_data = {}
    for name, q in frame_queues.items():
        try: 
            frames_data[name] = q.get_nowait()
        except queue.Empty:
            frames_data[name] = placeholder
            
    # --- tach frame va fps ---
    f1_tuple = frames_data.get("Cam 1", placeholder)
    f2_tuple = frames_data.get("Cam 2", placeholder)
    f3_tuple = frames_data.get("Cam 3", placeholder)
    f4_tuple = frames_data.get("Cam 4", placeholder)
    
    f1, fps1 = f1_tuple
    f2, fps2 = f2_tuple
    f3, fps3 = f3_tuple
    f4, fps4 = f4_tuple
    
    # --- ve fps len frame ---
    cv2.putText(f1, f"Cam 1: {fps1:.2f} FPS", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(f2, f"Cam 2: {fps2:.2f} FPS", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(f3, f"Cam 3: {fps3:.2f} FPS", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(f4, f"Cam 4: {fps4:.2f} FPS", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # ---ghep khung hinh ---
    top_row = np.hstack((f1, f2))
    bottom_row = np.hstack((f3, f4))
    
    combined_frame = np.vstack((top_row, bottom_row))
    
    # --- tinh toan va ve fps cua vong lap chinh ---
    display_fps_frame_count += 1
    current_time = time.time()
    elapsed_display_fps_time = current_time - display_fps_start_time

    if elapsed_display_fps_time >= 1.0:
        display_fps = display_fps_frame_count / elapsed_display_fps_time
        display_fps_frame_count = 0
        display_fps_start_time = current_time

    fps_text = f"Display loop FPS: {display_fps:.2f} (Target: {TARGET_FPS})"
    cv2.rectangle(combined_frame, (30, 30), (400, 40), (0, 0, 0), -1)
    cv2.putText(combined_frame, fps_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Multi-Camera Display", combined_frame)
    
    loop_elapsed_time = time.time() - loop_start_time
    sleep_time = max(0, loop_delay - loop_elapsed_time)
    
    if cv2.waitKey(int(sleep_time * 1000)) & 0xFF == ord('q'):
        break
    
    print("Dang thoat...")
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()