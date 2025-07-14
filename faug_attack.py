import argparse
import os
import pickle
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from data_loader import ImageNetDataset
import tqdm
import transferattack
from transferattack.utils import *
from config import attack_config
from models.model_utils import get_aug_model_by_name, get_model_by_name, get_attack_by_name, create_save_path, ConfigParser, obj


def save_images(images, filenames, save_dir):
    for i, batch in enumerate(zip(images, filenames)):
        img, path = batch
        dest_path = os.path.join(save_dir, path)
        transforms.ToPILImage()(img).save(dest_path)


def test(model_ori, model, atk, data_loader, save_dir, **atk_config):
    total = 0
    fr_num = 0
    acc_num = 0
    acc_clean_num = 0
    acc_adv_num = 0
    # 针对特定网络生成的对抗样本，在其他网络上进行验证
    for i, batch in enumerate(data_loader):
        images, labels, filenames = batch
        images, labels = images.to(device), labels.to(device)

        f_clean_pred = torch.argmax(model_ori(images), dim=1)
        perturbations = atk(images, f_clean_pred.detach(), **atk_config)
        # perturbations = atk.forward(images, f_clean_pred.detach(), **atk_config)

        adv_images = torch.clamp(images + perturbations, 0, 1)

        f_clean_adv = torch.argmax(model_ori(adv_images), dim=1)
        fr_num += (f_clean_pred != f_clean_adv).sum().item()
        acc_num += (f_clean_pred == f_clean_adv).sum().item()

        acc_clean_num += (f_clean_pred == labels).sum().item()
        acc_adv_num += (f_clean_adv == labels).sum().item()

        total += adv_images.shape[0]

        save_images(adv_images, filenames, save_dir)

    acc_clean_avg = round(100 * acc_clean_num / total, 2)
    acc_adv_avg = round(100 * acc_adv_num / total, 2)
    fooling_rate = round(100 * fr_num / total, 2)
    acc = round(100 * acc_num / total, 2)

    return acc_clean_avg, acc_adv_avg, fooling_rate, acc


def get_args():
    parser = argparse.ArgumentParser(description="Args Container")
    parser.add_argument("--data-path", type=str, default='/mnt/jfs/wangdonghua/dataset/NIPS2017-ImageNet1K/')
    parser.add_argument("--save-path", type=str, default='./saved_images/Compared_method_FAUG')
    parser.add_argument("--timestamp", type=str, default='')
    parser.add_argument("--eps", type=float, default=16.0)
    parser.add_argument("--batch_size", type=int, default=16)
    # admix smm naa taig cwa
    parser.add_argument('-e', '--eval', action='store_true', help='attack/evluation')
    parser.add_argument('--attack', default='naa', type=str, help='the attack algorithm', choices=transferattack.attack_zoo.keys())
    parser.add_argument('--epoch', default=10, type=int, help='the iterations for updating the adversarial patch')
    # parser.add_argument('--batchsize', default=32, type=int, help='the bacth size')
    # parser.add_argument('--eps', default=16 / 255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--alpha', default=2.0 / 255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--momentum', default=1.0, type=float, help='the decay factor for momentum based attack')
    parser.add_argument('--model', default='resnet50', type=str, help='the source surrogate model')
    parser.add_argument('--ensemble', action='store_true', default=False, help='enable ensemble attack')
    parser.add_argument('--random_start', default=False, type=bool, help='set random start')
    # parser.add_argument('--input_dir', default='./data', type=str, help='the path for custom benign images, default: untargeted attack data')
    # parser.add_argument('--output_dir', default='./results', type=str, help='the path to store the adversarial patches')
    parser.add_argument('--targeted', action='store_true', default=False, help='targeted attack')
    parser.add_argument('--GPU_ID', default='0', type=str)

    parser.add_argument("--aug-type", type=str, default='NOISE')
    parser.add_argument("--layer", type=str, default=None, help="vgg6： feture_07, resnet50: layer2")
    parser.add_argument("--noise-type", type=str, default='NORMAL', help="NORMAL, UNIFORM")
    parser.add_argument("--std1", type=float, default=None)
    parser.add_argument("--cfg", type=str, default='resnet50.yaml',
                        help="resnet50 vgg19_bn resnext50_32x4d densenet121 pit_b_224 swin_s3_base_224 vit_base_patch16_224 visformer_small")
    parser.add_argument("--atk-cfg", type=str, default='attacks.yaml')
    args = parser.parse_args()
    return args

naa_feature_layers = {
    "resnet50": "layer1",
    "resnext50_32x4d": "layer1",
    "vgg19_bn": "features_0",
    "densenet121": "features_conv0",
    "visformer_small": "stage3_0",
    "pit_b_224": "blocks_1_0",
    "vit_base_patch16_224": "blocks_5",
    "swin_s3_base_224": "layers_0",
}

if __name__ == "__main__":
    """
    admix smm naa cwa taig
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = get_args()
    time_str = time.strftime("%m-%d-%H-%M", time.localtime())
    args.timestamp = time_str
    args.cfg = "./configs/" + args.cfg
    args.atk_cfg = "./configs/" + args.atk_cfg

    save_dir = create_save_path(args)  # "Attack-model_Model-Name_Epsilon_aug-type_time"
    args.save_path = save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    config = ConfigParser(args.cfg)
    # atk_settings = ConfigParser(args.atk_cfg)
    # .NORMAL
    model = get_aug_model_by_name(args.model, device=device)
    model_ori = get_model_by_name(args.model, device=device)
    # atk = get_attack_by_name(args.atk_name, model, **(eval(f"atk_config.{args.atk_name}").__dict__))
    atk_config = attack_config[args.attack]

    # atk_config['model_name'] = model
    atk_config['targeted'] = args.targeted
    atk_config['epsilon'] = args.eps / 255
    atk_config['alpha'] = args.alpha
    atk_config['epoch'] = args.epoch

    for k, v in eval(f"config.{args.aug_type}").__dict__.items():

        if isinstance(v, obj):
            for kk, vv in v.__dict__.items():
                atk_config[kk] = vv
        else:
            atk_config[k] = v
    if args.attack == "naa":
        atk_config['feature_layer'] = naa_feature_layers[args.model]
        
    # attacker = transferattack.load_attack_class(args.attack)(
    #     model_name=model,
    #     targeted=args.targeted,
    #     epsilon=args.eps / 255
    # )
    attacker = transferattack.load_attack_class(args.attack)(model_name=model, **atk_config)
    # del atk_config['model_name']
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = ImageNetDataset(root=args.data_path, transform=test_transform)
    # train_set, test_set = torch.utils.data.random_split(dataset, [990, 10])
    data_loader = DataLoader(dataset, batch_size=args.batch_size)
    acc_clean, acc_adv, fooling_rate, acc = test(model_ori, model, attacker, data_loader, save_dir, **atk_config)
    print(f"The mertric of {os.path.basename(save_dir)} is: "
                  f"\nACC_clean: {acc_clean}"
                  f"\nACC_adv: {acc_adv}"
                  f"\nACC(after attack): {acc}"
                  f"\nFooling rate: {fooling_rate}")
    print("#"*15, "Done", "#"*15)