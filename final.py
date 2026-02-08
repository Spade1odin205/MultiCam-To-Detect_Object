import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
from threading import Thread
import math

# =============================
# 1. CẤU HÌNH HỆ THỐNG
# =============================
# Bạn có thể thêm bao nhiêu cam tùy thích vào đây (4, 8, 12, 16...)
CAM_SOURCES = [
    "rtsp://admin:IFPREC@192.168.1.114:554/ch1/main",
    # "rtsp://admin:CPSFLT@192.168.1.118:554/ch1/main",
    # "rtsp://admin:DVCLRQ@192.168.1.121:554/ch1/main",
    # "rtsp://admin:DTAJVP@192.168.1.124:554/ch1/main",
    # "rtsp://admin:NXKPHU@192.168.1.157:554/ch1/main",
    # "rtsp://admin:HLTHKD@192.168.1.127:554/ch1/main",
    # "rtsp://admin:NWKGIC@192.168.1.135:554/ch1/main",
    # "rtsp://admin:WSLRQC@192.168.1.128:554/ch1/main",
    # "rtsp://admin:UAQHDA@192.168.1.132:554/ch1/main",
    # "rtsp://admin:NNFVAJ@192.168.1.137:554/ch1/main",
    # "rtsp://admin:YDVFNP@192.168.1.133:554/ch1/main",
    # "rtsp://admin:TIJEQB@192.168.1.134:554/ch1/main",
    # "rtsp://admin:CYXJBA@192.168.1.138:554/ch1/main",
    # "rtsp://admin:EIUSAY@192.168.1.143:554/ch1/main",
    # "rtsp://admin:VZBRIC@192.168.1.146:554/ch1/main",
    # "rtsp://admin:XLRPZQ@192.168.1.154:554/ch1/main"
]

ENGINE_PATH = "/home/odin/BTL_AI/models/best_mosaic_fp16.engine"

# Kích thước Model AI (Cố định theo file engine đã train)
MODEL_W, MODEL_H = 640, 640

# Kích thước 1 ô nhỏ trong Mosaic (Model 640 chia đôi là 320)
SUB_W, SUB_H = MODEL_W // 2, MODEL_H // 2

# Kích thước nhận từ Camera (Input gốc)
CAM_W, CAM_H = 1280, 720 

# Kích thước ô hiển thị trên màn hình
# Giảm xuống 320x180 để khi hiển thị 8-16 cam không bị tràn màn hình
GRID_CELL_W, GRID_CELL_H = 320, 180 

CONF_THRESH = 0.4
IOU_THRESH = 0.45
SKIP_FRAMES = 2 # Tăng lên 1 hoặc 2 nếu thấy lag khi chạy > 8 cam

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# =============================
# 2. HÀM VẼ & TIỆN ÍCH
# =============================
def draw_text(img, text, pos, color=(255, 255, 255), bg_color=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (t_w, t_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x, y - t_h - 2), (x + t_w, y + 2), bg_color, -1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness)

# =============================
# 3. CLASS CAMERA
# =============================
class ThreadedCamera:
    def __init__(self, source, id):
        self.id = id
        if isinstance(source, str) and source.startswith("rtsp"):
            # Dùng tcp để ổn định hình ảnh hơn khi chạy nhiều cam
            self.pipeline = (
                f"rtspsrc location={source} latency=200 protocols=udp ! "
                f"rtph265depay ! h265parse ! nvv4l2decoder ! "
                f"nvvidconv ! video/x-raw, format=BGRx, width={CAM_W}, height={CAM_H} ! "
                f"videoconvert ! video/x-raw, format=BGR ! "
                f"appsink drop=true sync=false max-buffers=1"
            )
            self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        else:
            self.cap = cv2.VideoCapture(source)

        self.status = False
        self.frame = None
        self.running = True
        self.fps = 0
        self.frame_cnt = 0
        self.start_t = time.time()
        
        # Biến lưu kết quả detect riêng cho cam này
        self.detections = []

        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            if self.cap.isOpened():
                (self.status, self.frame) = self.cap.read()
                if self.status:
                    self.frame_cnt += 1
                    diff = time.time() - self.start_t
                    if diff >= 1.0:
                        self.fps = self.frame_cnt / diff
                        self.frame_cnt = 0
                        self.start_t = time.time()
                else:
                    time.sleep(0.01) # Sleep ngắn hơn để check nhanh hơn
            else:
                time.sleep(0.1)
            time.sleep(0.005)

    def get_frame(self):
        return self.status, self.frame, self.fps

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

# =============================
# 4. TENSORRT SETUP
# =============================
def load_engine(path):
    print(f"Loading engine: {path}")
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

try:
    engine = load_engine(ENGINE_PATH)
    context = engine.create_execution_context()
except Exception as e:
    print(f"❌ Lỗi tải Engine: {e}")
    exit()

input_idx = engine.get_binding_index("images")
output_idx = engine.get_binding_index("output0")
if input_idx == -1: input_idx = 0
if output_idx == -1: output_idx = 1
input_shape = engine.get_binding_shape(input_idx)
output_shape = engine.get_binding_shape(output_idx)

d_input = cuda.mem_alloc(trt.volume(input_shape) * np.float32().nbytes)
d_output = cuda.mem_alloc(trt.volume(output_shape) * np.float32().nbytes)
bindings = [int(d_input), int(d_output)]
stream = cuda.Stream()

# =============================
# 5. XỬ LÝ MOSAIC (GỘP & TÁCH)
# =============================

def preprocess_mosaic(frame_list):
    """
    Gộp danh sách ảnh (tối đa 4) vào lưới 2x2.
    Trả về Tensor input và danh sách index hợp lệ cục bộ (0-3).
    """
    mosaic_img = np.full((MODEL_H, MODEL_W, 3), 114, dtype=np.uint8)
    
    positions = [(0, 0), (SUB_W, 0), (0, SUB_H), (SUB_W, SUB_H)]
    valid_local_indices = []

    # Duyệt tối đa 4 ảnh trong list đưa vào
    for i, frame in enumerate(frame_list):
        if i >= 4: break 
        if frame is None: continue

        img_resized = cv2.resize(frame, (SUB_W, SUB_H))
        x_off, y_off = positions[i]
        mosaic_img[y_off:y_off+SUB_H, x_off:x_off+SUB_W] = img_resized
        valid_local_indices.append(i)

    # Chuẩn hóa
    img_input = cv2.cvtColor(mosaic_img, cv2.COLOR_BGR2RGB)
    img_input = img_input.astype(np.float32) / 255.0
    img_input = img_input.transpose((2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0)
    img_input = np.ascontiguousarray(img_input)
    
    return img_input, valid_local_indices

def postprocess_mosaic(output, valid_local_indices):
    """
    Xử lý đầu ra, trả về dict {local_index: detections}
    """
    predictions = np.reshape(output, (output_shape[1], output_shape[2])).T
    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    
    mask = scores > CONF_THRESH
    boxes = boxes[mask]
    scores = scores[mask]
    
    if len(boxes) == 0: return {}

    xc, yc, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1, y1, x2, y2 = xc - w/2, yc - h/2, xc + w/2, yc + h/2
    
    final_boxes = np.stack((x1, y1, x2 - x1, y2 - y1), axis=1).tolist()
    indices = cv2.dnn.NMSBoxes(final_boxes, scores.tolist(), CONF_THRESH, IOU_THRESH)
    
    results = {i: [] for i in valid_local_indices}
    
    scale_x = CAM_W / SUB_W
    scale_y = CAM_H / SUB_H

    if len(indices) > 0:
        for i in indices.flatten():
            bx, by, bw, bh = final_boxes[i]
            score = scores[i]
            
            cx, cy = bx + bw/2, by + bh/2
            cam_local_idx = -1
            offset_x, offset_y = 0, 0

            # Xác định vị trí trong lưới 2x2
            if cx < SUB_W and cy < SUB_H:       cam_local_idx = 0; offset_x, offset_y = 0, 0
            elif cx >= SUB_W and cy < SUB_H:    cam_local_idx = 1; offset_x, offset_y = SUB_W, 0
            elif cx < SUB_W and cy >= SUB_H:    cam_local_idx = 2; offset_x, offset_y = 0, SUB_H
            elif cx >= SUB_W and cy >= SUB_H:   cam_local_idx = 3; offset_x, offset_y = SUB_W, SUB_H
            
            if cam_local_idx in results:
                local_x = bx - offset_x
                local_y = by - offset_y
                
                # Scale về kích thước gốc 1280x720
                orig_x1 = local_x * scale_x
                orig_y1 = local_y * scale_y
                orig_x2 = (local_x + bw) * scale_x
                orig_y2 = (local_y + bh) * scale_y
                
                results[cam_local_idx].append(([orig_x1, orig_y1, orig_x2, orig_y2], score))

    return results

def create_grid(images, cell_w, cell_h):
    count = len(images)
    if count == 0: return None
    # Tự động tính số cột (4 cột là đẹp cho 8 hoặc 16 cam)
    cols = 4 if count >= 4 else count
    rows = math.ceil(count / cols)
    
    grid_rows = []
    blank = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    idx = 0
    
    for r in range(rows):
        row_imgs = []
        for c in range(cols):
            if idx < count:
                small_img = cv2.resize(images[idx], (cell_w, cell_h), interpolation=cv2.INTER_LINEAR)
                row_imgs.append(small_img)
                idx += 1
            else:
                row_imgs.append(blank)
        grid_rows.append(np.hstack(row_imgs))
    
    return np.vstack(grid_rows)

# =============================
# 6. MAIN LOOP
# =============================
cameras = []
print(f"⏳ Đang kết nối {len(CAM_SOURCES)} cameras...")

# --- THAY ĐỔI: Không giới hạn số lượng cam nữa ---
for i, src in enumerate(CAM_SOURCES):
    cam = ThreadedCamera(src, i)
    cameras.append(cam)
    # time.sleep(0.5) # Bỏ hoặc giảm sleep để khởi động nhanh hơn

print(f"✅ Đã kết nối {len(cameras)} cameras. Chế độ: MULTI-BATCH MOSAIC.")

sys_fps = 0
sys_frame_cnt = 0
sys_start_t = time.time()
frame_loop_cnt = 0

try:
    while True:
        # Tính FPS
        sys_frame_cnt += 1
        curr_time = time.time()
        if curr_time - sys_start_t >= 1.0:
            sys_fps = sys_frame_cnt / (curr_time - sys_start_t)
            sys_frame_cnt = 0
            sys_start_t = curr_time

        # 1. Thu thập Frame từ TẤT CẢ camera
        all_frames = []
        for cam in cameras:
            status, frame, _ = cam.get_frame()
            if status and frame is not None:
                all_frames.append(frame)
            else:
                all_frames.append(None) 
        
        # Check skip frame
        run_inference = (frame_loop_cnt % (SKIP_FRAMES + 1) == 0)
        
        # 2. XỬ LÝ "CUỐN CHIẾU" (Time-Multiplexing)
        # Duyệt qua từng nhóm 4 camera: [0-3], [4-7], [8-11]...
        if run_inference:
            for i in range(0, len(cameras), 4):
                # Lấy ra 4 frame của nhóm hiện tại
                batch_frames = all_frames[i : i+4]
                
                # Nếu không có ảnh nào thì bỏ qua
                if not any(f is not None for f in batch_frames): continue

                # A. Tạo Mosaic cho nhóm này
                inp, valid_local_indices = preprocess_mosaic(batch_frames)
                
                # B. Inference (Vẫn dùng model Batch=1, chạy tuần tự từng nhóm)
                cuda.memcpy_htod_async(d_input, inp, stream)
                context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                output = np.empty(output_shape, dtype=np.float32)
                cuda.memcpy_dtoh_async(output, d_output, stream)
                stream.synchronize()
                
                # C. Tách kết quả
                group_results = postprocess_mosaic(output, valid_local_indices)
                
                # D. Gán kết quả về đúng Camera Object
                # local_idx (0-3) + i (offset nhóm) = Global ID
                for local_idx, detections in group_results.items():
                    global_cam_id = i + local_idx
                    
                    if global_cam_id < len(cameras):
                        cam_obj = cameras[global_cam_id]
                        if detections:
                            best_det = max(detections, key=lambda x: x[1])
                            cam_obj.detections = [best_det]
                        else:
                            cam_obj.detections = []

        # 3. Vẽ và Hiển thị
        display_list = []
        for i, cam in enumerate(cameras):
            if cam.frame is not None:
                # Vẽ detection
                for box, score in cam.detections:
                    x1, y1, x2, y2 = map(int, box)
                    # Clip coordinates
                    x1, y1 = max(0, min(x1, CAM_W)), max(0, min(y1, CAM_H))
                    x2, y2 = max(0, min(x2, CAM_W)), max(0, min(y2, CAM_H))
                    
                    cv2.rectangle(cam.frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    draw_text(cam.frame, f"{score:.2f}", (x1, y1-10), (0,255,0))
                
                info_text = f"CAM {cam.id} | FPS: {int(cam.fps)}"
                draw_text(cam.frame, info_text, (20, 40), color=(0, 255, 255))
                display_list.append(cam.frame)
            else:
                blank = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
                cv2.putText(blank, f"CAM {cam.id} LOST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                display_list.append(blank)

        if display_list:
            final_view = create_grid(display_list, GRID_CELL_W, GRID_CELL_H)
            draw_text(final_view, f"SYS FPS: {int(sys_fps)} | Cams: {len(cameras)}", (20, 30), color=(0, 0, 255), bg_color=(255, 255, 255))
            cv2.imshow("Multi-Camera System", final_view)

        frame_loop_cnt += 1
        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    print("Stopping...")

finally:
    for c in cameras: c.stop()
    cv2.destroyAllWindows()
    try:
        del context
        del engine
    except:
        pass
    print("Closed.")
