import cv2
import numpy as np

# Load image
img_path = "/Users/thanhhuongtran/Documents/him/seed-size/seed_ground_truth/raw_images/Anh2.jpg"
img = cv2.imread(img_path)

if img is None:
    print(f"Error: Unable to load image from {img_path}")
    exit()

# Convert to grayscale and enhance contrast
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_gray = clahe.apply(gray)

# Apply adaptive thresholding
binary = cv2.adaptiveThreshold(
    enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)

# Morphological operations
kernel = np.ones((3, 3), np.uint8)
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

# Find and filter contours
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

filtered_contours = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h

    if area>5000:
        filtered_contours.append(cnt)

# Draw bounding boxes
output_image = img.copy()
for cnt in filtered_contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Detected Seeds", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
