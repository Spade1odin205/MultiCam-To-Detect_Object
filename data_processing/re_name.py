import os

def rename_files(directory, text_to_add):
    # Kiểm tra xem đường dẫn thư mục có tồn tại không
    if not os.path.exists(directory):
        print(f"Thư mục '{directory}' không tồn tại. Vui lòng kiểm tra lại.")
        return

    # Duyệt qua từng file trong thư mục
    count = 0
    for filename in os.listdir(directory):
        # Lấy phần tên và phần đuôi mở rộng (extension)
        name, ext = os.path.splitext(filename)
        
        # Chỉ xử lý file .txt và .jpg (không phân biệt hoa thường)
        if ext.lower() in ['.txt', '.jpg']:
            
            # Tạo tên mới: Tên cũ + Ký hiệu thêm vào + Đuôi file
            new_name = f"{name}{text_to_add}{ext}"
            
            # Tạo đường dẫn đầy đủ cũ và mới
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            
            # Thực hiện đổi tên
            try:
                os.rename(old_path, new_path)
                print(f"Đã đổi: {filename} -> {new_name}")
                count += 1
            except Exception as e:
                print(f"Lỗi khi đổi tên file {filename}: {e}")

    print(f"\nHoàn tất! Đã đổi tên {count} file.")

# --- CẤU HÌNH ---
# 1. Điền đường dẫn thư mục chứa file của bạn vào đây:
folder_path = r'D:\Code\Python\Project\Multi_cam\data_processing_btl\Loi-20251208T095853Z-1-001\Loi' 

# 2. Điền ký hiệu bạn muốn thêm vào:
symbol = "_v1" 

# Chạy hàm
rename_files(folder_path, symbol)