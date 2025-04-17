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


IMAGE_PATH = '/teamspace/studios/this_studio/some code/seed_ground_truth/raw_images/Anh1.jpg'
PRETRAINED_WEIGHT = '/teamspace/studios/this_studio/some code/unet_model.h5'
image_size = (256, 256)  # Resize ảnh về kích thước cố định
# Đường kính thực tế của đồng xu
REAL_COIN_DIAMETER_MM = 21.0  # 21mm

def load_image_infer(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, image_size)
    img = np.array(img, dtype=np.float32)/255
    return img[None,...]  # just expand dims, lazy 


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
    return None  # Nếu không tìm thấy đồng xu lớn hơn min_size

REAL_COIN_DIAMETER_MM = 21.0  # 21mm
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

def visualize_results_optimized(image, mask, dimensions, image_path, min_size=5, max_size=30):
    """
    Trực quan hóa các hạt lúa với độ rõ ràng hơn:
    - Lọc bỏ các đối tượng quá lớn hoặc quá nhỏ.
    - Hiển thị chữ không bị chồng lấn.

    Args:
        image: Ảnh gốc (numpy array).
        mask: Mask dự đoán (numpy array).
        dimensions: Danh sách các kích thước [(length, width), ...].
        min_size: Kích thước tối thiểu để hiển thị (đơn vị pixel).
        max_size: Kích thước tối đa để hiển thị (đơn vị pixel).
    """
    # Sao chép ảnh gốc để vẽ
    output_image = image.copy()

    # Tìm contours từ mask
    contours, _ = cv2.findContours((mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = -1
    widths = []
    lengths = []
    for i, contour in enumerate(contours):
        # Lấy kích thước của đối tượng
        length, width = dimensions[i]

        # Bỏ qua các đối tượng quá lớn hoặc quá nhỏ
        if length < min_size or width < min_size or length > max_size or width > max_size:
            continue
        count += 1
        widths.append(width)
        lengths.append(length)
        # Vẽ contour
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(output_image, [box], 0, (0, 255, 0), 1)  # Độ dày đường = 1

        # Vẽ chú thích cách ô một khoảng nhỏ
        x, y = int(rect[0][0]), int(rect[0][1]) - 10  # Đặt text phía trên đối tượng
        cv2.putText(output_image, f"{count}",
                    (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1, cv2.LINE_AA)

    # Hiển thị ảnh kết quả
    # plt.figure(figsize=(10, 10))
    # plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    # plt.title("Optimized Visualization")
    # plt.axis("off")
    # plt.show()
    image_name = image_path.split('/')[-1].split('.')[0]
    # plt.savefig(f'{image_name}_rice_seed_predict.png')
    cv2.imwrite(f'{image_name}_rice_seed_predict.png', output_image)
    # Save to file:
    df_result = pd.DataFrame({'width': widths, 'height': lengths})
    df_result.to_csv(f'{image_name}_rice_seed_dimensions.csv', index=True, index_label='index')


def unet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)

    u3 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c4)
    u3 = concatenate([u3, c3])
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(u3)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    u2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u2 = concatenate([u2, c2])
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(u2)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    u1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u1 = concatenate([u1, c1])
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(u1)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

if __name__ == '__main__':
    model = unet_model()
    model.load_weights(PRETRAINED_WEIGHT)
    image = load_image_infer(IMAGE_PATH)
    mask = model.predict(image)

    image = (image[0] * 255).astype(np.uint8)
    mask = mask[0, :, :, 0]
    dimensions = calculate_dimensions(mask)
    visualize_results_optimized(image, mask, dimensions, IMAGE_PATH)