import argparse
import os
import cv2
import numpy as np
import pandas as pd
import math
from ultralytics import YOLO

# Hàm chuyển polygon đồng xu sang bounding box
def coin_polygon_to_bbox(coin_polygon):
    if coin_polygon is None or (isinstance(coin_polygon, np.ndarray) and coin_polygon.size == 0) or len(coin_polygon) == 0:
        return None
    coin_polygon = np.array(coin_polygon)
    min_x = np.min(coin_polygon[:, 0])
    min_y = np.min(coin_polygon[:, 1])
    max_x = np.max(coin_polygon[:, 0])
    max_y = np.max(coin_polygon[:, 1])
    return (min_x, min_y, max_x, max_y)

# === Hàm sử dụng Otsu thresholding + minAreaRect để tính kích thước hạt ===
def calculate_seed_sizes_using_minAreaRect(image, seed_boxes, mm_per_pixel):
    seed_sizes_mm = []
    
    for i, box in enumerate(seed_boxes):
        x1, y1, x2, y2 = map(int, box)
        roi = image[y1:y2, x1:x2]  # Cắt vùng ảnh chứa hạt thóc

        # Chuyển ảnh về grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Áp dụng threshold Otsu để tách hạt khỏi nền
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Tìm contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)  # Chọn contour lớn nhất
            rect = cv2.minAreaRect(largest_contour)  # Bounding box xoay
            (w, h) = rect[1]  # Lấy width, height từ box xoay

            length_px = max(w, h)
            width_px = min(w, h)

            # Chuyển đổi pixel → mm
            seed_sizes_mm.append((width_px * mm_per_pixel, length_px * mm_per_pixel))
        else:
            print(f"Hạt {i+1}: Không tìm thấy contour!")
            seed_sizes_mm.append((None, None))

    return seed_sizes_mm

# === Hàm vẽ bounding box và hiển thị kích thước ===
def visualize_and_save(image_path, seed_boxes, seed_sizes_mm, coin_bbox, output_dir):
    image = cv2.imread(image_path)
    ids, widths, lengths = [], [], []

    # Vẽ bounding box đồng xu (nếu có)
    if coin_bbox is not None:
        min_x, min_y, max_x, max_y = map(int, coin_bbox)
        cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
        cv2.putText(image, "Coin", (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Vẽ bounding box hạt thóc + hiển thị kích thước
    for i, (box, size) in enumerate(zip(seed_boxes, seed_sizes_mm)):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        length_mm, width_mm = size
        if length_mm is not None and width_mm is not None:
            text = f"{i+1}: {round(length_mm, 2)}x{round(width_mm, 2)} mm"
            ids.append(i+1)
            widths.append(round(width_mm, 2))
            lengths.append(round(length_mm, 2))
        else:
            text = f"{i+1}: Không đo được"
            ids.append(i+1)
            widths.append(None)
            lengths.append(None)

        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Lưu ảnh kết quả và CSV
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    annotated_path = os.path.join(output_dir, f"{image_name}_annotated.jpg")
    csv_path = os.path.join(output_dir, f"{image_name}_dimensions.csv")

    cv2.imwrite(annotated_path, image)
    pd.DataFrame({'ID': ids, 'Chiều rộng (mm)': lengths, 'Chiều dài (mm)': widths}).to_csv(csv_path, index=False)

    print(f"Annotated image saved to: {annotated_path}")
    print(f"Dimensions saved to: {csv_path}")

# === Chương trình chính ===
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rice Seed Analysis with YOLOv8")
    parser.add_argument("--image-path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the YOLOv8 model file (e.g., best.pt).")
    parser.add_argument("--project-dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--ground-size", type=float, required=True, help="Diameter of the reference coin in mm.")
    args = parser.parse_args()

    # Tạo thư mục lưu kết quả
    os.makedirs(args.project_dir, exist_ok=True)

    # Load mô hình YOLOv8
    model = YOLO(args.model_path)

    # Chạy suy luận trên ảnh
    results = model(args.image_path)[0]  # Lấy kết quả ảnh đầu tiên

    # Lấy danh sách bounding box & polygon
    seed_boxes = []
    coin_polygon = None

    for i, r in enumerate(results.boxes):
        cls_id = int(r.cls[0]) if hasattr(r.cls, '__iter__') else int(r.cls)
        label = results.names[cls_id].lower()
        box = r.xyxy.cpu().numpy()[0]
        if label == "seed":
            seed_boxes.append(box)
    

    # Lấy polygon của đồng xu từ results.masks.xy (nếu có)
    if results.masks and results.masks.xy:
        coin_polygon = np.array(results.masks.xy[0])

    # Chuyển polygon đồng xu sang bounding box
    coin_bbox = coin_polygon_to_bbox(coin_polygon)
    if coin_bbox is None:
        print("Không tìm thấy đồng xu trong ảnh! Không thể tính kích thước hạt thóc.")
        exit()

    # Tính kích thước đồng xu theo pixel
    min_x, min_y, max_x, max_y = coin_bbox
    coin_diameter_px = max(max_x - min_x, max_y - min_y)
    mm_per_pixel = args.ground_size / coin_diameter_px

    # Tính kích thước hạt thóc theo `cv2.minAreaRect()` + Otsu
    image = cv2.imread(args.image_path)
    seed_sizes_mm = calculate_seed_sizes_using_minAreaRect(image, seed_boxes, mm_per_pixel)

    # Lưu kết quả và vẽ
    visualize_and_save(args.image_path, seed_boxes, seed_sizes_mm, coin_bbox, args.project_dir)
