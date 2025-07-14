"""对抗比赛用--by wdh 20241018"""
import os
import matplotlib.pyplot as plt
from PIL import Image


def get_image_resolution(image_path):
    """获取图像的分辨率"""
    try:
        img = Image.open(image_path)
        return img.size
    except IOError:
        print(f"无法打开文件：{image_path}")
        return None


def plot_resolution_scatter(directory, save_name="examples", is_save=False):
    """绘制图像分辨率的散点图"""
    widths = []
    heights = []

    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            filepath = os.path.join(directory, filename)

            # 获取图像分辨率
            resolution = get_image_resolution(filepath)
            if resolution:
                width, height = resolution
                widths.append(width)
                heights.append(height)

    # 绘制散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(widths, heights, alpha=0.5)
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title('Image Resolution Scatter Plot')
    if is_save:
        plt.savefig(f"{save_name}.png", dpi=300)
    # plt.show()

def plot_resolution_scatter(directory, save_name="examples", is_save=False):
    """绘制图像分辨率的散点图"""
    widths = []
    heights = []

    # 遍历目录中的所有文件
    for foldername in os.listdir(directory):
        folder_dir = os.path.join(directory, foldername)
        for filename in os.listdir(folder_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                filepath = os.path.join(folder_dir, filename)

                # 获取图像分辨率
                resolution = get_image_resolution(filepath)
                if resolution:
                    width, height = resolution
                    widths.append(width)
                    heights.append(height)

    # 绘制散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(widths, heights, alpha=0.5)
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title('Image Resolution Scatter Plot')
    if is_save:
        plt.savefig(f"{save_name}.png", dpi=300)
    plt.show()


# 数据集所在目录
# directory = '/mnt2/datasets/attack_dataset/det_dataset/FullDataSet/AllImages'  # 替换为你的数据集目录
# # 绘制散点图
# plot_resolution_scatter(directory, save_name="det_all_images", is_save=True)

# 数据集所在目录
directory = '/mnt2/datasets/attack_dataset/cls_dataset'  # 替换为你的数据集目录
# 绘制散点图
plot_resolution_scatter(directory, save_name="cls_all_images", is_save=True)