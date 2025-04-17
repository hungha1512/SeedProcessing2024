import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
import os

# TensorFlow GPU setup
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(
                gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
            )
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

# Constants
image_size = (256, 256)  # Fixed input size for model
OBJECT_THRESHOLD = 0.5
PRETRAINED_WEIGHT = "src/main/java/org/uet/rislab/seed/applicationlinux/pythoncore/unetpp_model.h5"
# Load image for inference
def load_image_infer(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    original_size = img.shape[:2][::-1]  # (width, height)
    img_resized = cv2.resize(img, image_size)
    img_resized = np.array(img_resized, dtype=np.float32) / 255
    return img_resized[None, ...], original_size, img

# Calculate coin diameter
def calculate_coin_diameter(mask, min_size, object_threshold):
    contours, _ = cv2.findContours((mask > object_threshold).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        diameter = max(width, height)
        if diameter > min_size:
            return diameter
    return None

# Calculate dimensions
def calculate_dimensions(mask, original_size, real_coin_diameter_mm, mask_threshold=OBJECT_THRESHOLD, min_coin_size=30):
    dimensions = []
    contours, _ = cv2.findContours((mask >= mask_threshold).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        dimensions.append((max(width, height), min(width, height)))

    coin_dia = calculate_coin_diameter(mask, min_coin_size, mask_threshold)
    if coin_dia is None:
        print(f"WARNING: Coin diameter too small. Adjust parameters.")
        return dimensions

    dimensions_mm = [(w * real_coin_diameter_mm / coin_dia, h * real_coin_diameter_mm / coin_dia) for w, h in dimensions]
    return dimensions_mm

# Visualize results
def visualize_results_optimized(image, mask, dimensions, image_path, image_analysis_dir, result_dir, original_size, original_image, min_size=1, max_size=16):
    output_image = original_image.copy()
    mask_resized = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours((mask_resized > OBJECT_THRESHOLD).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ids, widths, lengths = [], [], []
    id = 0
    for i, contour in enumerate(contours):
        if i >= len(dimensions):
            continue
        length, width = dimensions[i]
        if length < min_size or width < min_size or length > max_size or width > max_size:
            continue
        widths.append(width)
        lengths.append(length)

        id += 1
        ids.append(id)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(output_image, [box], 0, (0, 255, 0), 2)

        x, y = int(rect[0][0]), int(rect[0][1])  # Center of the rectangle
        label_id = f"{id}"
        cv2.putText(
            output_image, label_id, (x, y - 10),  # Slightly above the center
            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA
        )

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    annotated_path = os.path.join(image_analysis_dir, f"{image_name}_annotated.jpg")
    csv_path = os.path.join(result_dir, f"{image_name}_dimensions.csv")

    cv2.imwrite(annotated_path, output_image)
    pd.DataFrame({'ID': ids,'Chiều rộng (mm)': widths, 'Chiều dài (mm)': lengths}).to_csv(csv_path, index=False)
    print(f"Annotated image saved to: {annotated_path}")
    print(f"Dimensions saved to: {csv_path}")

def conv_block(x, filters, kernel_size=3, activation='relu', padding='same'):
    x = Conv2D(filters, kernel_size, activation=activation, padding=padding)(x)
    x = Conv2D(filters, kernel_size, activation=activation, padding=padding)(x)
    return x


# Khối upsample & concat (giống U-Net)
def up_concat_block(x, skip, filters):
    x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
    x = concatenate([x, skip])
    x = conv_block(x, filters)
    return x


def unet_plus_plus_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)

    # --------------------- ENCODER ---------------------
    # block 1
    c1_0 = conv_block(inputs, 64)  # X_1,0
    p1_0 = MaxPooling2D((2, 2))(c1_0)

    # block 2
    c2_0 = conv_block(p1_0, 128)  # X_2,0
    p2_0 = MaxPooling2D((2, 2))(c2_0)

    # block 3
    c3_0 = conv_block(p2_0, 256)  # X_3,0
    p3_0 = MaxPooling2D((2, 2))(c3_0)

    # block 4
    c4_0 = conv_block(p3_0, 512)  # X_4,0
    p4_0 = MaxPooling2D((2, 2))(c4_0)

    # bottleneck
    c5_0 = conv_block(p4_0, 1024)  # X_5,0 (bottleneck)

    # --------------------- DECODER: STAGE 1 ---------------------
    # Mỗi tầng up từ bottleneck/đầu ra trước và concat với encoder
    c4_1 = up_concat_block(c5_0, c4_0, 512)  # X_4,1
    c3_1 = up_concat_block(c4_0, c3_0, 256)  # X_3,1
    c2_1 = up_concat_block(c3_0, c2_0, 128)  # X_2,1
    c1_1 = up_concat_block(c2_0, c1_0, 64)  # X_1,1

    # --------------------- DECODER: STAGE 2 (++ connections) ---------------------
    # Kết hợp thêm skip của stage 1 + skip gốc encoder
    x4_2_input = concatenate([c4_1, c4_0])
    c4_2 = up_concat_block(c5_0, x4_2_input, 512)

    x3_2_input = concatenate([c3_1, c3_0])
    c3_2 = up_concat_block(c4_1, x3_2_input, 256)

    x2_2_input = concatenate([c2_1, c2_0])
    c2_2 = up_concat_block(c3_1, x2_2_input, 128)

    x1_2_input = concatenate([c1_1, c1_0])
    c1_2 = up_concat_block(c2_1, x1_2_input, 64)

    # Output cuối (chọn c1_2)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c1_2)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Main script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rice Seed Analysis")
    parser.add_argument("--image-path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--project-dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--ground-size", type=float, required=True, help="Diameter of the reference object in mm.")
    args = parser.parse_args()

    output_dir = args.project_dir
    os.makedirs(output_dir, exist_ok=True)

    image_analysis_dir = os.path.join(output_dir, "Image_analysis")
    result_dir = os.path.join(output_dir, "Result")

    model = unet_plus_plus_model()
    model.load_weights(PRETRAINED_WEIGHT)
    image, original_size, original_image = load_image_infer(args.image_path)
    mask = model.predict(image)[0, :, :, 0]

    dimensions = calculate_dimensions(mask, original_size, args.ground_size)
    visualize_results_optimized(image, mask, dimensions, args.image_path, image_analysis_dir, result_dir, original_size, original_image)
