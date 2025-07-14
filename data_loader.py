import os
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd

def default_fn(img_path):
    return Image.open(img_path).convert("RGB")

class ImageNetDataset(Dataset):
    def __init__(self, root, transform=None, default_fn=default_fn):
        self.transform = transform
        self.default_fn = default_fn
        datas = []
        data_dir = root + "images"
        csv_files = root + "images.csv"

        # data_dir = root
        # csv_files = "D:/DataSource/ImageNet-NIPS2017/images.csv"

        df = pd.read_csv(csv_files, index_col=None)
        for i, item in enumerate(df.iterrows()):
            filename = item[1]["ImageId"]
            src_label = item[1]["TrueLabel"] - 1
            tgt_label = item[1]["TargetClass"] - 1
            datas.append((os.path.join(data_dir, f"{filename}.png"), src_label, tgt_label))
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        img_path, src_label, _ = self.datas[index]

        image = self.default_fn(img_path)
        filename = os.path.basename(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, src_label, filename

class ImageNetDatasetAnalysis(Dataset):
    def __init__(self, root, transform=None, default_fn=default_fn):
        self.transform = transform
        self.default_fn = default_fn
        datas = []
        # data_dir = root + "images"
        # csv_files = root + "images.csv"

        data_dir = root
        csv_files = "D:/DataSource/ImageNet-NIPS2017/images.csv"

        df = pd.read_csv(csv_files, index_col=None)
        for i, item in enumerate(df.iterrows()):
            filename = item[1]["ImageId"]
            src_label = item[1]["TrueLabel"] - 1
            tgt_label = item[1]["TargetClass"] - 1
            datas.append((os.path.join(data_dir, f"{filename}.png"), src_label, tgt_label))
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        img_path, src_label, _ = self.datas[index]

        image = self.default_fn(img_path)
        filename = os.path.basename(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, src_label, filename

class ImageNetTargetDataset(Dataset):
    def __init__(self, root, transform=None, default_fn=default_fn):
        self.transform = transform
        self.default_fn = default_fn
        datas = []
        data_dir = root + "images"
        csv_files = root + "images.csv"
        df = pd.read_csv(csv_files, index_col=None)
        for i, item in enumerate(df.iterrows()):
            filename = item[1]["ImageId"]
            src_label = item[1]["TrueLabel"] - 1
            tgt_label = item[1]["TargetClass"] - 1
            datas.append((os.path.join(data_dir, f"{filename}.png"), src_label, tgt_label))
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        img_path, src_label, tgt_label = self.datas[index]

        image = self.default_fn(img_path)
        filename = os.path.basename(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, src_label, tgt_label, filename

class ImageNetDatasetEvaluate(Dataset):
    def __init__(self, root, adv_root, transform=None, default_fn=default_fn):
        self.transform = transform
        self.default_fn = default_fn
        self.adv_root = adv_root
        datas = []
        data_dir = root + "images"
        csv_files = root + "images.csv"
        df = pd.read_csv(csv_files, index_col=None)
        for i, item in enumerate(df.iterrows()):
            filename = item[1]["ImageId"]
            src_label = item[1]["TrueLabel"]
            tgt_label = item[1]["TargetClass"]
            datas.append((os.path.join(data_dir, f"{filename}.png"), src_label, tgt_label))
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        img_path, src_label, _ = self.datas[index]

        image = self.default_fn(img_path)
        filename = os.path.basename(img_path)
        adv_img = self.default_fn(os.path.join(self.adv_root, filename))
        if self.transform is not None:
            image = self.transform(image)
            adv_img = self.transform(adv_img)
        # return image, adv_img
        return image, adv_img, os.path.basename(img_path)

