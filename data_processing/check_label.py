import os

LABEL_DIR = r"D:\Code\Python\Project\Multi_cam\data_new\dataset_split_1\train"   # đường dẫn tới thư mục label

files_with_multiple_boxes = []

for label_file in os.listdir(LABEL_DIR):
    if not label_file.endswith(".txt"):
        continue

    path = os.path.join(LABEL_DIR, label_file)

    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) >= 2:
        files_with_multiple_boxes.append((label_file, len(lines)))

# In kết quả
print("Các file có từ 2 bounding box trở lên:")
for file, count in files_with_multiple_boxes:
    print(f"{file} -> {count} boxes")

print(f"\nTổng số file cần kiểm tra: {len(files_with_multiple_boxes)}")
