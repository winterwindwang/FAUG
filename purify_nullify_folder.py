import os
import shutil

project_dir="/mnt/jfs/wangdonghua/pythonpro/FeatureAug/Compared_method"

for folder in os.listdir(project_dir):
    save_path = os.path.join(project_dir, folder)
    file_list = os.listdir(save_path)
    if len(file_list) == 0:
        if os.path.exists(save_path):
            try:
                shutil.rmtree(save_path)
            except OSError as e:
                print("Error occurred while deleting the directory: ", e.strerror)
