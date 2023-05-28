import os
import random
from PIL import ImageEnhance, Image


def is_image_file(file_path):
    # 檢查檔案的副檔名是否屬於圖片格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    _, file_ext = os.path.splitext(file_path)
    return file_ext.lower() in image_extensions


def adjust_brightness(image_path, output_dir, brightness_factor):
    # 開啟圖片並調整亮度
    image = Image.open(image_path)
    enhancer = ImageEnhance.Brightness(image)
    adjusted_image = enhancer.enhance(brightness_factor)

    # 取得圖片檔名和副檔名
    image_name, image_ext = os.path.splitext(os.path.basename(image_path))

    # 組合新的圖片檔名
    new_image_name_1 = f"{image_name}_1{image_ext}"
    new_image_name_2 = f"{image_name}_2{image_ext}"

    # 設定輸出路徑
    output_path_1 = os.path.join(output_dir, new_image_name_1)
    output_path_2 = os.path.join(output_dir, new_image_name_2)

    # 儲存調整後的圖片
    adjusted_image.save(output_path_1)
    adjusted_image.save(output_path_2)


def random_brightness(input_dir, output_dir):
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)

    # 讀取輸入資料夾中的所有檔案
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    for file_name in files:
        # 取得檔案的完整路徑
        file_path = os.path.join(input_dir, file_name)

        # 檢查檔案是否為圖片檔案
        if not is_image_file(file_path):
            continue

        # 隨機生成亮度調整因子，範圍在0.5到1.5之間
        brightness_factor = random.uniform(0.5, 1.5)

        # 調整亮度並儲存圖片到輸出目錄
        adjust_brightness(file_path, output_dir, brightness_factor)


# 設定輸入和輸出目錄
input_directory = "C:/ml/new_data"
output_directory = "C:/ml/new_img"

# 呼叫函式來執行亮度調整並儲存圖片
random_brightness(input_directory, output_directory)
