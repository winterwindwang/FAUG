import argparse
import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from data_loader import ImageNetDatasetEvaluate
from model_utils import Normalize, get_model_by_name, logger_banner, logger_msg
import pandas as pd
import timm
import time


def save_images(images, filenames, save_dir):
    for i, batch in enumerate(zip(images, filenames)):
        img, path = batch
        dest_path = os.path.join(save_dir, path)
        transforms.ToPILImage()(img).save(dest_path)


@torch.no_grad()
def test_batch(model_list, data_loader):
    results = {}
    # 针对特定网络生成的对抗样本，在其他网络上进行验证
    for i, batch in enumerate(data_loader):
        images, adv_images, filenames = batch
        images, adv_images = images.to(device), adv_images.to(device)

        for model_name, model in model_list.items():

            if model_name not in results:
                results[model_name] = 0
            f_clean_pred = model(images)
            f_clean_adv = model(adv_images)
            fooling_num = (torch.argmax(f_clean_pred, dim=1) != torch.argmax(f_clean_adv, dim=1)).sum().item()
            results[model_name] += fooling_num

    for key, value in results.items():
        print(f"Fooling rate of {key} is: {round(value * 100/ 1000, 2)}")
    fr = [round(value * 100/ 1000, 2) for value in list(results.values())]
    print(fr)
    return results

@torch.no_grad()
def test_single(model, data_loader):

    total = 0
    fr_num = 0
    acc_num = 0
    # 针对特定网络生成的对抗样本，在其他网络上进行验证
    for i, batch in enumerate(data_loader):
        images, adv_images, filenames = batch
        images, adv_images = images.to(device), adv_images.to(device)

        f_clean_pred = torch.argmax(model(images), dim=1)
        f_clean_adv = torch.argmax(model(adv_images), dim=1)

        fr_num += (f_clean_pred != f_clean_adv).sum().item()
        acc_num += (f_clean_pred == f_clean_adv).sum().item()

        total += adv_images.shape[0]


    fooling_rate = round(100 * fr_num / total, 2)
    equal_rate = round(100 * acc_num / total, 2)

    return fooling_rate, equal_rate


def get_model(model_names, device):
    """
    adversarial trained model: https://drive.google.com/drive/folders/10cFNVEhLpCatwECA6SPB-2g0q5zZyfaw
    """
    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    model_list = {}
    for mn in model_names:
        model = get_model_by_name(mn, device=device)
        model = torch.nn.Sequential(normalize, model)
        model.eval()
        model.to(device)
        model_list[mn] = model
    return model_list

def get_args():
    parser = argparse.ArgumentParser(description="Args Container")
    parser.add_argument("--data_dir", type=str, default='/home/wdh/Datasets/ImageNet-NIPS2017/')
    parser.add_argument("--adv_data_dir", type=str, default='/mnt/jfs/wangdonghua/dataset/ImageNet/')
    parser.add_argument("--save_dir", type=str, default='saved_images/Compared_method')
    parser.add_argument("--model_name", type=str, default='')
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    return args

def data_dict(project_dir="/mnt/jfs/wangdonghua/pythonpro/FeatureAug"):
    file_dict = {

        # Analysis: influence of clean model and noise model
        'resnet50_MIFGSM_16': f'{project_dir}/saved_images/Compared_method/MIFGSM_resnet50_EPS16.0_02-17-11-00',
    }
    return file_dict
if __name__ == "__main__":
    args = get_args()
    file_dict = data_dict('F:/PythonPro/FeatureAug')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])

    model_names_list = "resnet50 densenet121 vgg19_bn resnext50_32x4d visformer_small pit_b_224 vit_base_patch16_224 swin_s3_base_224".split(' ')

    logger_banner("Begin Evaluation")
    logger_msg("Evaluation Models", model_names_list)

    data_frame_fooling_rate = pd.DataFrame(data=None, columns=["File", *(model_names_list)])
    data_frame_equal_rate = pd.DataFrame(data=None, columns=["File", *(model_names_list)])

    for key, adv_path in file_dict.items():
        logger_msg(key, adv_path)
        args.adv_data_dir = adv_path
        dataset = ImageNetDatasetEvaluate(root=args.data_dir, adv_root=args.adv_data_dir, transform=test_transform)
        data_loader = DataLoader(dataset, batch_size=args.batch_size)

        fooling_rate_list = []
        equal_rate_list = []
        for model_name in model_names_list:
            args.model_name = model_name
            model = get_model_by_name(args.model_name, device=device)
            fooling_rate, equal_rate = test_single(model, data_loader)
            fooling_rate_list.append(fooling_rate)
            equal_rate_list.append(equal_rate)
        logger_msg("Fooling rate", fooling_rate_list)
        logger_msg("Equal rate", equal_rate_list)
    #     data_frame_fooling_rate = data_frame_fooling_rate._append(pd.DataFrame([fooling_rate_list.insert(0, key)],
    #                                                                           columns=data_frame_fooling_rate.columns))
    #     data_frame_equal_rate = data_frame_equal_rate._append(pd.DataFrame([equal_rate_list.insert(0, key)],
    #                                                                       columns=data_frame_equal_rate.columns))
    # time_str = time.strftime("%m-%d-%H-%M", time.localtime())
    # save_dir = f"{args.save_dir}/Evaluation_result_{time_str}"
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    # data_frame_fooling_rate.to_csv(os.path.join(save_dir, "fooling_rate_result.csv"))
    # data_frame_equal_rate.to_csv(os.path.join(save_dir, "data_frame_equal_rate.csv"))
    # logger_msg("save_dir", save_dir)
    logger_banner("END")