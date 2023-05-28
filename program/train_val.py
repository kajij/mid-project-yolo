import os
import shutil
from collections import defaultdict
import random

# 設定原始檔案資料夾路徑和新檔案儲存資料夾路徑
original_folder = "C:/ml/data"
new_train_folder = "C:/ml/train_data"
new_val_folder = "C:/ml/val_data"

# 確定新檔案儲存資料夾存在
if not os.path.exists(new_train_folder):
    os.makedirs(new_train_folder)
if not os.path.exists(new_val_folder):
    os.makedirs(new_val_folder)

# 獲取原始檔案資料夾中的所有檔案
file_names = os.listdir(original_folder)

# 用 defaultdict 建立一個以檔名為 key，以該檔案的所有副檔名為 value 的字典
file_dict = defaultdict(list)
for file_name in file_names:
    file_name_without_ext, ext = os.path.splitext(file_name)
    file_dict[file_name_without_ext].append(ext)

# 迭代處理每個檔案
for file_name_without_ext, ext_list in file_dict.items():
    # 只處理檔名相同的包含jpg和txt的檔案
    if ".jpg" not in ext_list or ".txt" not in ext_list:
        continue
    
    # 隨機分配檔案到訓練集或驗證集
    if random.random() < 0.8:
        new_folder = new_train_folder
    else:
        new_folder = new_val_folder
    
    # 複製檔案到新的資料夾中
    for ext in ext_list:
        original_file_path = os.path.join(original_folder, file_name_without_ext + ext)
        new_file_path = os.path.join(new_folder, file_name_without_ext + ext)
        shutil.copyfile(original_file_path, new_file_path)
