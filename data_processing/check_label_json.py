import os
import json

LABEL_DIR = r"D:\Code\Python\Project\Multi_cam\data_new\frame\lan4\ngoc"

files_with_multiple_boxes = []

for file in os.listdir(LABEL_DIR):
    if not file.endswith(".json"):
        continue

    path = os.path.join(LABEL_DIR, file)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    shapes = data.get("shapes", [])
    if len(shapes) >= 2:
        files_with_multiple_boxes.append((file, len(shapes)))

print("File có từ 2 bounding box trở lên:")
for f, n in files_with_multiple_boxes:
    print(f"{f} -> {n} boxes")
