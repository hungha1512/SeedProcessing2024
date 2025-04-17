import argparse
import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import math

# === Hàm tính khoảng cách Euclidean ===
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# === Hàm tính kích thước hạt từ MASK ===
def calculate_seed_sizes_from_mask(image, masks, mm_per_pixel):
    seed_sizes_mm = []
    
    for mask in masks:
        mask = np.array(mask, dtype=np.int32)

        # Tạo binary mask
        mask_image = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_image, [mask], 255)

        # Tìm contours
        contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)  # Chọn contour lớn nhất
            rect = cv2.minAreaRect(largest_contour)  # Bounding box xoay
            (w, h) = rect[1]
            length_px = max(w, h)
            #box = cv2.boxPoints(rect)  # 4 góc của bounding box
            #box = np.int0(box)

            # Chiều dài = khoảng cách Euclidean giữa hai đỉnh xa nhất của bbox
            #length_px = euclidean_distance(box[0], box[2])

            # Tìm tâm của object
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            else:
                centroid = (0, 0)

            # Tìm điểm gần nhất với tâm theo phương vuông góc
            min_dist = float("inf")
            nearest_point = None
            for pt in mask:
                dist = euclidean_distance(centroid, pt)
                if dist < min_dist:
                    min_dist = dist
                    nearest_point = pt
            
            width_px = min_dist * 2  # Chiều rộng là 2 lần khoảng cách đến điểm gần nhất

            # Chuyển đổi pixel → mm
            seed_sizes_mm.append((length_px * mm_per_pixel, width_px * mm_per_pixel))  # Lưu đúng thứ tự chiều dài trước, chiều rộng sau
        else:
            seed_sizes_mm.append((None, None))

    return seed_sizes_mm

# === Hàm vẽ kết quả ===
def visualize_and_save(image_path, seed_masks, seed_sizes_mm, coin_bbox, output_dir):
    image = cv2.imread(image_path)
    ids, widths, lengths = [], [], []

    # Vẽ bounding box đồng xu (nếu có)
    if coin_bbox is not None:
        min_x, min_y, max_x, max_y = map(int, coin_bbox)
        cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
        cv2.putText(image, "Coin", (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Vẽ mask hạt thóc + hiển thị kích thước
    for i, (mask, size) in enumerate(zip(seed_masks, seed_sizes_mm)):
        cv2.polylines(image, [np.array(mask, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

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

        centroid = np.mean(mask, axis=0).astype(int)
        cv2.putText(image, text, tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Lưu ảnh kết quả và CSV
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    annotated_path = os.path.join(output_dir, f"{image_name}_annotated.jpg")
    csv_path = os.path.join(output_dir, f"{image_name}_dimensions.csv")

    cv2.imwrite(annotated_path, image)
    pd.DataFrame({'ID': ids, 'Chiều dài (mm)': widths, 'Chiều rộng (mm)': lengths}).to_csv(csv_path, index=False)

# === Chương trình chính ===
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rice Seed Analysis with YOLOv8 (Segmentation)")
    parser.add_argument("--image-path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the YOLOv8 model file (e.g., best.pt).")
    parser.add_argument("--project-dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--ground-size", type=float, required=True, help="Diameter of the reference coin in mm.")
    args = parser.parse_args()

    os.makedirs(args.project_dir, exist_ok=True)

    model = YOLO(args.model_path)
    results = model(args.image_path)[0]

    seed_masks = []
    coin_polygon = None

    if results.masks and results.masks.xy:
        for mask in results.masks.xy:
            seed_masks.append(mask)

    if results.masks and results.masks.xy:
        coin_polygon = np.array(results.masks.xy[0])

    coin_bbox = None
    if coin_polygon is not None:
        min_x, min_y = np.min(coin_polygon, axis=0)
        max_x, max_y = np.max(coin_polygon, axis=0)
        coin_bbox = (min_x, min_y, max_x, max_y)

    if coin_bbox is None:
        print("Không tìm thấy đồng xu trong ảnh! Không thể tính kích thước hạt thóc.")
        exit()

    min_x, min_y, max_x, max_y = coin_bbox
    coin_diameter_px = max(max_x - min_x, max_y - min_y)
    mm_per_pixel = args.ground_size / coin_diameter_px

    image = cv2.imread(args.image_path)
    seed_sizes_mm = calculate_seed_sizes_from_mask(image, seed_masks, mm_per_pixel)

    visualize_and_save(args.image_path, seed_masks, seed_sizes_mm, coin_bbox, args.project_dir)
