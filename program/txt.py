import os

# 設定原始檔案資料夾路徑和新檔案儲存資料夾路徑
original_folder = "C:/ml/new_data"
new_folder = "C:/ml/new/new_txt"

# 確定新檔案儲存資料夾存在
if not os.path.exists(new_folder):
    os.makedirs(new_folder)

# 獲取原始檔案資料夾中的所有檔案
file_names = os.listdir(original_folder)

# 迭代處理每個檔案
for file_name in file_names:
    # 確定檔案為txt檔
    if not file_name.endswith(".txt"):
        continue
    
    # 讀取檔案內容
    with open(os.path.join(original_folder, file_name), "r") as f:
        file_content = f.read()
    
    # 產生兩份新檔案
    for i in range(2):
        # 複製檔案內容
        new_file_content = file_content
        
        # 儲存新檔案
        new_file_name = "{}_{}.txt".format(os.path.splitext(file_name)[0], i+1)
        with open(os.path.join(new_folder, new_file_name), "w") as f:
            f.write(new_file_content)
