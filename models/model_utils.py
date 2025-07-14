import os
import timm
import torch
import numpy as np
from torch.nn import functional as F
import torchattacks
# from attacks import SSIFGSM, AMVIFGSM
# from models import *
import models
import sys
# sys.path.append("../attacks/")
# from attacks import baseline_attacks

# _CKPT_DIR = "/mnt/jfs/wangdonghua/torch/checkpoints"
_CKPT_DIR = "/mnt/jfs/wangdonghua/torch/checkpoints"

MODEL_CKPT_DICT = {
    "resnet50": "resnet50-19c8e357.pth",
    "densenet121": "densenet121-a639ec97.pth",
    "vgg19_bn": "vgg19_bn-c79401a0.pth",
    "resnext50_32x4d": "resnext50_32x4d-7cdf4587.pth",
    "visformer_small": "visformer_small_in1k.bin",
    "pit_b_224": "pit_b_224_in1k.bin",
    "vgg16": "vgg16-397923af.pth",
    # "vit_base_patch16_224": "vit_base_patch16_224.pth",
    "vit_base_patch16_224": "jx_vit_base_p16_224-80ecf9dd.pth",
    "swin_s3_base_224": "s3_b-a1e95db4.pth",
    "ens_adv_inception_resnet_v2": "ens_adv_inception_resnet_v2-2592a550.pth",
    "adv_inception_v3": "adv_inception_v3-9e27bd63.pth",
    "resnetv2_152x2_bit": "adv_inception_v3-9e27bd63.pth",
    "resnet152": "resnet152-394f9c45.pth",
    "inception_resnet_v2": "inception_resnet_v2-940b1cd6.pth",
}


def get_attack_by_name_v1(atk_name, model, **kwargs):
    attack = eval(f"baseline_attacks.{atk_name}")(model, **kwargs)
    return attack


def get_attack_by_name(atk_name, model, **kwargs):
    if atk_name in ["SSIFGSM", "AMVIFGSM"]:
        attack = eval(f"{atk_name}")(model, **kwargs)
    else:
        # attack = eval(f"baseline_attacks.{atk_name}")(model, **kwargs)
        attack = eval(f"torchattacks.{atk_name}")(model, **kwargs)
    return attack

def get_ens_attack_by_name(atk_name, model, **kwargs):
    attack = eval(f"baseline_attacks.{atk_name}")(model, **kwargs)
    # attack = eval(f"torchattacks.{atk_name}")(model, **kwargs)
    return attack


def get_model_by_name(model_name, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    try:
        # if model_name in ["visformer_small", "pit_b_224"]:
        #     model = eval(f"timm.create_model('{model_name}', checkpoint_path='{_CKPT_DIR}/{model_name}_in1k.bin')")
        # elif model_name in ["vit_base_patch16_224"]:
        #     # model = eval(f"timm.create_model('{model_name}', checkpoint_path='{_CKPT_DIR}/{model_name}.pth')")
        #     model = eval(f"timm.create_model('{model_name}', checkpoint_path='{_CKPT_DIR}/jx_vit_base_p16_224-80ecf9dd.pth')")
        # elif model_name in ["swin_s3_base_224"]:
        #     model = eval(f"timm.create_model('{model_name}', pretrained=True)")
        # else:
        #     model = eval(f"timm.create_model('{model_name}.tv_in1k', pretrained=True)")
        if model_name in ["adv_inception_v3", "ens_adv_inception_resnet_v2", "inception_resnet_v2"]:
            pretrained_cfg = {"file": f'{_CKPT_DIR}/{MODEL_CKPT_DICT[model_name]}', "num_classes": 1001}
        else:
            pretrained_cfg = {"file": f'{_CKPT_DIR}/{MODEL_CKPT_DICT[model_name]}'}
        model = eval(f"timm.create_model('{model_name}', pretrained=True, pretrained_cfg={pretrained_cfg})")
    except:
        raise ValueError(f"Unsupport {model_name} now!!")
    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    model = torch.nn.Sequential(normalize, model)
    model.to(device)
    model.eval()
    return model


def get_aug_model_by_name(model_name, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    try:
        pretrained_cfg = {"file": f'{_CKPT_DIR}/{MODEL_CKPT_DICT[model_name]}'}
        model = eval(f"models.{model_name}(pretrained=True, pretrained_cfg={pretrained_cfg})")
    except:
        raise ValueError(f"Unsupport {model_name} now!!")
    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    model = torch.nn.Sequential(normalize, model)
    model.to(device)
    model.eval()
    return model
    # if "resnet50" == model_name:
    #     model = resnet50(pretrained=True)
    # elif "resnet152" == model_name:
    #     model = resnet152(pretrained=True)
    # elif "vgg19_bn" == model_name:
    #     model = vgg19_bn(pretrained=True)
    # elif "inception_v3" == model_name:
    #     model = inception_v3(pretrained=True)
    # elif "resnet152" == model_name:
    #     model = inception_v4(pretrained=True)
    # elif "ens_adv_inception_resnet_v2" == model_name:
    #     model = ens_adv_inception_resnet_v2(pretrained=True)
    # elif "inception_resnet_v2" == model_name:
    #     model = inception_resnet_v2(pretrained=True)
    # else:
    #     raise ValueError(f"Unsupport {model_name} now!!")


def corrcoef(x):
    """传入一个tensor格式的矩阵x(x.shape(m,n))，输出其相关系数矩阵"""
    f = (x.shape[0] - 1) / x.shape[0]  # 方差调整系数
    x_reducemean = x - torch.mean(x, dim=0)
    numerator = torch.matmul(x_reducemean.T, x_reducemean) / x.shape[0]
    var_ = x.var(axis=0).reshape(x.shape[1], 1)
    denominator = torch.sqrt(torch.matmul(var_, var_.T)) * f
    corrcoef = numerator / (denominator + 1e-6)
    return corrcoef

class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def backward(self):
        return "mean={}, std={}".format(self.mean, self.std)

def normalize_fn(tensor, mean, std):
    """
    Differentiable version of functional.normalize
    - default assumes color channel is at dim = 1
    """
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


import yaml
import os


class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, obj(b) if isinstance(b, dict) else b)

class ConfigParser:
    def __init__(self, config_file):
        self.config_file = config_file
        self.load_config()

    def update_paramters(self, key, value):
        key_split = key.split(".")
        tmp_config = "self"

        for i in range(1, len(key_split)-1):
            if hasattr(eval(tmp_config), key_split[i]) and isinstance(eval(f"{tmp_config}.{key_split[i]}"), obj):
                tmp_config += f".{key_split[i]}"
        setattr(eval(tmp_config), key_split[-1], value)

    def load_config(self):
        cfg = yaml.load(open(self.config_file, encoding='utf-8'), Loader=yaml.FullLoader)
        for a, b in cfg.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, obj(b) if isinstance(b, dict) else b)

def update_config_with_args(args, config:ConfigParser):
    for k, v in args.__dict__.items():
        if k == "atk_name":
            if hasattr(config, v):
                config.update_paramters(f"config.{v}.eps", eval(f"config.{v}.eps / 255.0"))
        else:
            if k == "noise_type" and v == "NOISE":
                config.update_paramters(f"config.{args.aug_type}.{args.noise_type}", args.noise_type.lower())
            elif hasattr(args, "layer") and k == "layer" and v is not None:
                config.update_paramters(f"config.{args.aug_type}.layer", args.layer.lower())
            elif hasattr(args, "std1") and k == "std1" and v is not None:
                config.update_paramters(f"config.{args.aug_type}.NORMAL.std1", v)
            elif hasattr(args, "dropout_p") and k == "dropout_p" and v is not None:
                config.update_paramters(f"config.{args.aug_type}.dropout_p", v)

def logger_msg(k, v):
    msg ='{:>30} : {:<30}'.format(str(k), str(v))
    print(msg)
    return msg

def logger_banner(banner):
    dot = '------------------------------------------------------------'
    pos = int(len(dot) / 2 - len(banner) / 2)
    banner = dot[:pos] + banner + dot[pos+len(banner):]
    print(banner)

def logger_cfg(cfg, banner=None):
    if banner is not None:
        logger_banner(banner)
    for k, v in cfg.__dict__.items():
        if isinstance(v, obj):
            logger_cfg(v)
        else:
            logger_msg(k, v)

def logger(cfg, args, banner=None):
    if banner is not None:
        logger_banner(banner)
    else:
        logger_banner("Feature Augmentation")
    logger_msg("cfg", args.cfg)
    if hasattr(args, "aug_type"):
        logger_cfg(eval(f"cfg.{args.aug_type}"), "AUG-"+args.aug_type)
    elif hasattr(cfg, args.atk_name):
        logger_cfg(eval(f"cfg.{args.atk_name}"), "Attack-" + args.atk_name)
    else:
        logger_cfg(eval(f"cfg.{args.atk_name}"), "Attack-" + args.atk_name)
    logger_cfg(args, "args")
    logger_banner("END")


def create_save_path(args):
    folder_name = f"{args.attack.upper()}_{args.model}_EPS{args.eps}_{args.timestamp}"
    return os.path.join(args.save_path, folder_name)

def create_save_path_abla(args):
    if len(args.aug_type) == 0:
        folder_name = f"{args.atk_name.upper()}_{args.model_name}_EPS{args.eps}_{args.layer}_std1{args.std1}_{args.timestamp}"
    else:
        folder_name = f"{args.atk_name.upper()}_{args.model_name}_EPS{args.eps}_{args.layer}_std1{args.std1}_{args.aug_type}_{args.timestamp}"
    return os.path.join(args.save_path, folder_name)

def create_save_path_ens(args):
    if "," in args.model_name:
        model_name = args.model_name.replace(",", "_")
    else:
        model_name = args.model_name
    if len(args.aug_type)  == 0:
        folder_name = f"{args.atk_name.upper()}_{model_name}_EPS{args.eps}_{args.timestamp}"
    else:
        folder_name = f"{args.atk_name.upper()}_{model_name}_EPS{args.eps}_{args.aug_type}_{args.timestamp}"
    return os.path.join(args.save_path, folder_name)

def merge_multi_config(args, model_names, configs):
    merged_dict = dict()
    for i, (mname, cfg) in enumerate(zip(model_names, configs)):
        if mname not in merged_dict:
            merged_dict[mname] = eval(f'cfg.{args.aug_type}').__dict__
    return  merged_dict



def feature_process(feat, **kwargs):
    if "noise" in kwargs['process_type']:
        if kwargs['is_partial']:
            if kwargs['feat_sort_type']=='var':
                with torch.no_grad():
                    channel_diff_var = torch.var(feat.data, dim=(2,3))
                    _, sorted_index = torch.sort(channel_diff_var, dim=1, descending=True)
            elif kwargs['feat_sort_type']=='minmax':
                with torch.no_grad():
                    channel_diff = torch.empty((feat.shape[0], feat.shape[1]))
                    for i, img in enumerate(feat):
                        for j, c_z in enumerate(img):
                            channel_diff[i, j] = torch.abs(torch.max(c_z) - torch.min(c_z))
                    _, sorted_index = torch.sort(channel_diff, dim=1, descending=True)
            elif kwargs['feat_sort_type'] == 'channel_mean':
                with torch.no_grad():
                    channel_mean = torch.mean(torch.abs(feat), dim=[1, 2, 3], keepdim=True)
                    large_feat = (torch.abs(feat) >= channel_mean).sum(dim=[2, 3])
                    _, sorted_index = torch.sort(large_feat, dim=1, descending=True)
            else:
                raise ValueError(f"Unsupport {kwargs['feat_sort_type']} feat sort manner")

            noise = torch.zeros_like(feat)
            important_split = int(sorted_index.shape[1] * kwargs['partial']) # sorted_index.shape[1] // 2
            print("important_index", sorted_index[0])
            index_important = sorted_index[:, :important_split].unsqueeze(-1).\
                unsqueeze(-1).expand(-1,-1,feat.shape[-2], feat.shape[-1])
            index_less_important = sorted_index[:, important_split:].unsqueeze(-1). \
                unsqueeze(-1).expand(-1, -1, feat.shape[-2], feat.shape[-1])

            if kwargs['noise_type'] == "normal":
                noise_important = torch.zeros(*(index_important.shape), device=noise.device).normal_(
                    mean=kwargs['NORMAL'].mean1, std=kwargs['NORMAL'].std1)
                noise_less_important = torch.zeros(*(index_less_important.shape), device=noise.device).normal_(
                    mean=kwargs['NORMAL'].mean2, std=kwargs['NORMAL'].std2)
            elif kwargs['noise_type'] == "uniform":
                noise_important = torch.zeros(*(index_important.shape), device=noise.device).uniform_(kwargs['NORMAL'].lower1, kwargs['NORMAL'].upper1)
                noise_less_important = torch.zeros(*(index_less_important.shape), device=noise.device).uniform_(kwargs['NORMAL'].lower2, kwargs['NORMAL'].upper2)
            else:
                raise ValueError(f"Unsupport {kwargs['noise_type']} noise type now!!")
            noise.scatter_(1, index_important, noise_important)
            noise.scatter_(1, index_less_important, noise_less_important)
        else:
            if kwargs['noise_type'] == "normal":
                std = feat.std().item()
                mean = kwargs['mean1']
                noise = torch.zeros_like(feat).normal_(mean=mean, std=kwargs['std1'] * std)

                # noise = torch.zeros_like(feat).normal_(mean=kwargs['mean1'], std=kwargs['std1'])
            elif kwargs['noise_type'] == "uniform":
                noise = torch.zeros_like(feat).uniform_(kwargs['lower1'], kwargs['upper1'])
            else:
                raise ValueError(f"Unsupport {kwargs['noise_type']} noise type now!!")
    elif "extrapolation" in kwargs['process_type']:
        feat_ori = feat.clone()
        for i, fi in enumerate(feat):
            with torch.no_grad():
                feat_var = fi.detach().reshape(fi.shape[0], -1)
                feat_indexs = list(range(feat_var.shape[0]))
                corref = corrcoef(feat_var.T)
            for j, channel in enumerate(fi):
                cur_p = torch.softmax(corref[i][None,:], dim=1)[0]
                cur_p = cur_p.cpu().data.numpy()
                exchange_idx = np.random.choice(feat_indexs, size=1, p=cur_p)[0]
                feat.data[i][j] = (feat.data[i][j] - feat.data[i][exchange_idx]) * kwargs['lambda1'] + feat.data[i][j]
        noise = feat - feat_ori
    elif "dropout" in kwargs['process_type']:
        feat_ori = feat.clone()
        feat = F.dropout(feat, p=kwargs['dropout_p'])
        noise = feat - feat_ori
    elif "feature_dis" in kwargs['process_type']:
        feat_ori = feat.clone()
        for i, fi in enumerate(feat):  # 遍历样本
            for j, channel in enumerate(fi):  # 遍历通道特征
                src_shape = channel.shape
                channel = torch.abs(channel.reshape(1,-1).squeeze())
                _, sorted_index = torch.sort(channel, descending=True)
                mask = torch.zeros_like(channel)

                important_split = int(sorted_index.shape[0] * kwargs['partial'])  # sorted_index.shape[1] // 2
                mask[sorted_index[:important_split]] = 1
                mask = mask.reshape(src_shape)
                # 在重要特征上添加更加的噪声，在不重要的特征上添加更多的噪声
                if kwargs['noise_type'] == "normal":
                    feat.data[i][j] = (feat.data[i][j] + torch.ones_like(mask).normal_(mean=kwargs['NORMAL'].mean1, std=kwargs['NORMAL'].std1)) * mask \
                                  + (1 - mask) * (feat.data[i][j] + torch.ones_like(mask).normal_(mean=kwargs['NORMAL'].mean2, std=kwargs['NORMAL'].std2))
                elif kwargs['noise_type'] == "uniform":
                    feat.data[i][j] = (feat.data[i][j] + torch.ones_like(mask).uniform_(kwargs['NORMAL'].lower1, kwargs['NORMAL'].upper1)) * mask \
                                      + (1 - mask) * (feat.data[i][j] + torch.ones_like(mask).uniform_(kwargs['NORMAL'].lower2, kwargs['NORMAL'].upper2))
        noise = feat - feat_ori
    else:
        raise ValueError(f"Unsupport {kwargs['process_type']} feature augmentation!!")
    return noise