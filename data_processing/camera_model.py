import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
from threading import Thread

# =============================
# CONFIG
# =============================
CAM_SOURCES = [
    "rtsp://admin:DTAJVP@192.168.1.124:554/ch1/main",
    "rtsp://admin:EIUSAY@192.168.1.143:554/ch1/main",
    # Thêm các cam khác vào đây
]

# Path tới engine
ENGINE_PATH = "/home/odin/BTL_AI/models/yolov8n_fp16.engine"
INPUT_W = 640
INPUT_H = 640
CONF_THRESH = 0.45
IOU_THRESH = 0.45

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# =============================
# CLASS: CAMERA ĐA LUỒNG (ĐÃ TỐI ƯU GSTREAMER)
# =============================
class ThreadedCamera:
    def __init__(self, source, id):
        self.id = id
        if isinstance(source, str) and source.startswith("rtsp"):
            # --- PIPELINE TỐI ƯU ---
            # Resize trực tiếp về 640x640 trên GPU bằng nvvidconv
            self.pipeline = (
                f"rtspsrc location={source} latency=200 ! "
                f"rtph265depay ! h265parse ! nvv4l2decoder ! "
                f"nvvidconv ! video/x-raw, format=BGRx, width={INPUT_W}, height={INPUT_H} ! "
                f"videoconvert ! video/x-raw, format=BGR ! "
                f"appsink drop=true sync=false max-buffers=1"
            )
            self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        else:
            self.cap = cv2.VideoCapture(source)

        self.status = False
        self.frame = None
        self.running = True
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            if self.cap.isOpened():
                (self.status, self.frame) = self.cap.read()
            else:
                time.sleep(0.1)
            # Ngủ cực ngắn để nhường CPU cho luồng AI
            time.sleep(0.001) 

    def get_frame(self):
        return self.status, self.frame

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

# =============================
# LOAD ENGINE & PREPARE
# =============================
def load_engine(path):
    print(f"Loading engine: {path}")
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine(ENGINE_PATH)
context = engine.create_execution_context()

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
# HELPER FUNCTIONS (ĐÃ ĐƠN GIẢN HÓA VÌ ẢNH ĐÃ LÀ 640x640)
# =============================
def preprocess(img):
    # Vì GStreamer đã resize về 640x640 rồi, ta không cần cv2.resize nữa
    # Tuy nhiên, nếu ảnh gốc không vuông, GStreamer sẽ kéo dãn ảnh.
    # Để đơn giản và nhanh nhất, ta chấp nhận ảnh hơi méo và nạp thẳng vào model.
    
    # 1. Convert BGR -> RGB
    img_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 2. Normalize 0-1
    img_input = img_input.astype(np.float32) / 255.0
    
    # 3. HWC -> CHW
    img_input = img_input.transpose((2, 0, 1))
    
    # 4. Add Batch dimension
    img_input = np.expand_dims(img_input, axis=0)
    img_input = np.ascontiguousarray(img_input)
    
    return img_input

def postprocess(output, orig_w, orig_h):
    predictions = np.reshape(output, (output_shape[1], output_shape[2])).T
    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    
    mask = scores > CONF_THRESH
    boxes = boxes[mask]
    scores = scores[mask]
    
    if len(boxes) == 0: return []

    # Vì ảnh đầu vào GStreamer đã là 640x640 (khớp input model)
    # và ta hiển thị cũng trên khung 640x640 đó luôn
    # nên không cần scale tọa độ phức tạp.
    
    xc, yc, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = xc - w/2
    y1 = yc - h/2
    x2 = xc + w/2
    y2 = yc + h/2

    final_boxes = np.stack((x1, y1, x2 - x1, y2 - y1), axis=1).tolist()
    indices = cv2.dnn.NMSBoxes(final_boxes, scores.tolist(), CONF_THRESH, IOU_THRESH)
    
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            box = final_boxes[i]
            results.append(([box[0], box[1], box[0]+box[2], box[1]+box[3]], scores[i]))
    return results

# =============================
# MAIN EXECUTION
# =============================
cameras = []
print("⏳ Khởi tạo cameras...")
for i, src in enumerate(CAM_SOURCES):
    cam = ThreadedCamera(src, i)
    cameras.append(cam)
    time.sleep(1)

print("✅ Sẵn sàng!")
prev_time = 0

try:
    while True:
        curr_time = time.time()
        fps = 0
        if prev_time != 0: fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Ghép ảnh để hiển thị (tùy chọn)
        display_frames = []

        for cam in cameras:
            status, frame = cam.get_frame()
            
            if status and frame is not None:
                # Frame lúc này ĐÃ LÀ 640x640 nhờ GStreamer
                
                # --- Inference ---
                inp = preprocess(frame)
                cuda.memcpy_htod_async(d_input, inp, stream)
                context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                output = np.empty(output_shape, dtype=np.float32)
                cuda.memcpy_dtoh_async(output, d_output, stream)
                stream.synchronize()
                
                # --- Postprocess ---
                # Truyền vào 640, 640 vì frame hiển thị cũng là kích thước này
                detections = postprocess(output, INPUT_W, INPUT_H)

                # --- Vẽ ---
                for box, score in detections:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{score:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                cv2.putText(frame, f"CAM {cam.id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                display_frames.append(frame)
            else:
                # Ảnh đen nếu mất tín hiệu
                display_frames.append(np.zeros((INPUT_H, INPUT_W, 3), dtype=np.uint8))

        # Hiển thị FPS tổng
        print(f"\rTotal FPS Loop: {fps:.2f}", end="")

        # Hiển thị từng cam (hoặc bạn có thể ghép lại bằng np.hstack như file thread)
        for i, frame in enumerate(display_frames):
            cv2.imshow(f"Cam {i}", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    for c in cameras: c.stop()
    cv2.destroyAllWindows()
    del context
    del engine