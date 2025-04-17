# Xử lý dữ liệu: chuẩn bị ảnh và mask
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import glob
import pandas as pd


IMAGE_PATH = '/Users/thanhhuongtran/Documents/him/seed-size/dataset_close_full/images/train/_MG_0097.JPG'
PRETRAINED_WEIGHT = 'unetpp_model.h5'
image_size = (256, 256)  # Resize ảnh về kích thước cố định
target_size = (2560, 2560)

def load_image_infer(img_path):
    img = cv2.imread(img_path)
    # Store original size for later use
    original_size = img.shape[:2][::-1]  # (width, height)
    # Resize for model input
    img_resized = cv2.resize(img, image_size)
    img_resized = np.array(img_resized, dtype=np.float32)/255
    return img_resized[None,...], original_size, img  # Return original size and original image

def calculate_coin_diameter(mask, min_size=30, OBJECT_THRESHOLD=0.2):
    """
    Tính đường kính đồng xu từ mask.

    Args:
        mask: Mask nhị phân của ảnh.
        min_size: Kích thước tối thiểu để lọc đồng xu (để loại bỏ các hạt lúa nhỏ).

    Returns:
        float: Đường kính đồng xu (đơn vị: pixels).
    """
    contours, _ = cv2.findContours((mask > OBJECT_THRESHOLD).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        diameter = max(width, height)
        if diameter > min_size:  # Giả định đồng xu là đối tượng lớn nhất
            return diameter
    return None 

REAL_COIN_DIAMETER_MM = 20.0

def calculate_dimensions(mask, mask_threshold=0.5, min_coin_size=30):
  dimensions = []  # Initialize dimensions list here
  contours, _ = cv2.findContours((mask >= mask_threshold).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  for contour in contours:
    rect = cv2.minAreaRect(contour)
    width, height = rect[1]
    dimensions.append((max(width, height), min(width, height)))
  #  Giả định đồng xu là đối tượng lớn nhất
  coin_dia = calculate_coin_diameter(mask, OBJECT_THRESHOLD=0.2, min_size=min_coin_size)
  if coin_dia is None:
    print(f'WARNING! COIN DIAMETER= {coin_dia} < 30 PIXELS, SEED PIXEL, MASK THRESHOLD MAY DECREASE FOR BETTER RESULTS')
    return dimensions
  print(f"Coin dia: {coin_dia}")
  dimensions_mm = []
  for (w, h) in dimensions:
    dimensions_mm.append((w*REAL_COIN_DIAMETER_MM/coin_dia, h*REAL_COIN_DIAMETER_MM/coin_dia))
  return dimensions_mm

def visualize_results_optimized(image, mask, dimensions, image_path, original_size, original_image, min_size=5, max_size=30):
    """
    Args:
        image: Processed image array
        mask: Predicted mask
        dimensions: List of dimensions
        image_path: Path to image
        original_size: Original image dimensions (width, height)
        original_image: Original image without preprocessing
        min_size: Minimum size threshold
        max_size: Maximum size threshold
    """
    # Use original image instead of processed one
    output_image = original_image.copy()
    
    # Scale mask back to original size
    mask_resized = cv2.resize(mask, original_size)
    
    # Find contours on the resized mask
    contours, _ = cv2.findContours((mask_resized > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Scale factor for dimensions
    scale_x = original_size[0] / image_size[0]
    scale_y = original_size[1] / image_size[1]
    
    count = -1
    widths = []
    lengths = []
    for i, contour in enumerate(contours):
        length, width = dimensions[i]
        
        count += 1
        widths.append(width)
        lengths.append(length)
        
        # Draw contour
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(output_image, [box], 0, (0, 255, 0), 2)  # Increased thickness for better visibility
        
        # Scale text position and size based on image dimensions
        x, y = int(rect[0][0]), int(rect[0][1]) - int(20 * scale_y)  # Adjusted offset
        font_scale = min(scale_x, scale_y) * 0.4  # Adjust font size based on image scale
        cv2.putText(output_image, f"{count}",
                    (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 2, cv2.LINE_AA)

    # Save results
    image_name = image_path.split('/')[-1].split('.')[0]
    cv2.imwrite(f'{image_name}_rice_seed_predict.png', output_image)
    
    # Save dimensions to CSV
    df_result = pd.DataFrame({'width': widths, 'height': lengths})
    df_result.to_csv(f'{image_name}_rice_seed_dimensions.csv', index=True, index_label='index')

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


if __name__ == '__main__':
    #model = unet_model()
    model = unet_plus_plus_model()
    model.load_weights(PRETRAINED_WEIGHT)
    image, original_size, original_image = load_image_infer(IMAGE_PATH)
    mask = model.predict(image)

    image = (image[0] * 255).astype(np.uint8)
    mask = mask[0, :, :, 0]
    dimensions = calculate_dimensions(mask)
    visualize_results_optimized(image, mask, dimensions, IMAGE_PATH, original_size, original_image)