import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from torch.hub import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1, model_urls
import numpy as np
from .model_utils import feature_process


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    # def feature_process(self, feat, process_type, **kwargs):
    #
    #     if "noise" in process_type:
    #         if kwargs['is_partial']:
    #             if kwargs['feat_sort_type']=='var':
    #                 with torch.no_grad():
    #                     channel_diff_var = torch.var(feat.data, dim=(2,3))
    #                     _, sorted_index = torch.sort(channel_diff_var, dim=1, descending=True)
    #             elif kwargs['feat_sort_type']=='minmax':
    #                 with torch.no_grad():
    #                     channel_diff = torch.empty((feat.shape[0], feat.shape[1]))
    #                     for i, img in enumerate(feat):
    #                         for j, c_z in enumerate(img):
    #                             channel_diff[i, j] = torch.abs(torch.max(c_z) - torch.min(c_z))
    #                     _, sorted_index = torch.sort(channel_diff, dim=1, descending=True)
    #             elif kwargs['feat_sort_type'] == 'channel_mean':
    #                 with torch.no_grad():
    #                     channel_mean = torch.mean(torch.abs(feat), dim=[1, 2, 3], keepdim=True)
    #                     large_feat = (torch.abs(feat) >= channel_mean).sum(dim=[2, 3])
    #                     _, sorted_index = torch.sort(large_feat, dim=1, descending=True)
    #             else:
    #                 raise ValueError(f"Unsupport {kwargs['feat_sort_type']} feat sort manner")
    #
    #             noise = torch.zeros_like(feat)
    #             important_split = int(sorted_index.shape[1] * kwargs['partial']) # sorted_index.shape[1] // 2
    #             index_important = sorted_index[:, :important_split].unsqueeze(-1).\
    #                 unsqueeze(-1).expand(-1,-1,feat.shape[-2], feat.shape[-1])
    #             index_less_important = sorted_index[:, important_split:].unsqueeze(-1). \
    #                 unsqueeze(-1).expand(-1, -1, feat.shape[-2], feat.shape[-1])
    #
    #             if kwargs['noise_type'] == "normal":
    #                 noise_important = torch.zeros(*(index_important.shape), device=noise.device).normal_(
    #                     mean=kwargs['mean1'], std=kwargs['std1'])
    #                 noise_less_important = torch.zeros(*(index_less_important.shape), device=noise.device).normal_(
    #                     mean=kwargs['mean2'], std=kwargs['std2'])
    #             elif kwargs['noise_type'] == "uniform":
    #                 noise_important = torch.zeros(*(index_important.shape), device=noise.device).uniform_(kwargs['lower1'], kwargs['upper1'])
    #                 noise_less_important = torch.zeros(*(index_less_important.shape), device=noise.device).uniform_(kwargs['lower2'], kwargs['upper2'])
    #             else:
    #                 raise ValueError(f"Unsupport {kwargs['noise_type']} noise type now!!")
    #             noise.scatter_(1, index_important, noise_important)
    #             noise.scatter_(1, index_less_important, noise_less_important)
    #         else:
    #             if kwargs['noise_type'] == "normal":
    #                 noise = torch.zeros_like(feat).normal_(mean=kwargs['mean1'], std=kwargs['std1'])
    #             elif kwargs['noise_type'] == "uniform":
    #                 noise = torch.zeros_like(feat).uniform_(kwargs['lower1'], kwargs['upper1'])
    #             else:
    #                 raise ValueError(f"Unsupport {kwargs['noise_type']} noise type now!!")
    #     return noise

    def _forward_impl(self, x: Tensor, **kwargs) -> Tensor:
        # See note [TorchScript super()]

        x = self.conv1(x)
        if 'conv' in kwargs['layer']:
            if "begin_indicator" in kwargs and kwargs['begin_indicator']:
                noise = feature_process(x.data, **kwargs)
                x.data += noise
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)
        if 'maxpool' in kwargs['layer']:
            if "begin_indicator" in kwargs and kwargs['begin_indicator']:
                noise = feature_process(x.data, **kwargs)
                x.data += noise

        x = self.layer1(x)
        ################ Feature Aug ####################
        if 'layer1' in kwargs['layer']:
            if "begin_indicator" in kwargs and kwargs['begin_indicator']:
                noise = feature_process(x.data, **kwargs)
                x.data += noise
        ################ END ####################

        x = self.layer2(x)

        ################ Feature Aug ####################
        if 'layer2' in kwargs['layer']:
            if "begin_indicator" in kwargs and kwargs['begin_indicator']:
                # x.data += torch.ones_like(x).normal_(mean=0.0, std=0.2)  # best
                noise = feature_process(x.data, **kwargs)
                x.data += noise


        ################ END ####################


        x = self.layer3(x)
        ################ Feature Aug ####################
        if 'layer3' in kwargs['layer']:
            if "begin_indicator" in kwargs and kwargs['begin_indicator']:
                noise = feature_process(x.data, **kwargs)
                x.data += noise
        ################ END ####################

        x = self.layer4(x)

        ################ Feature Aug ####################
        if 'layer4' in kwargs['layer']:
            if "begin_indicator" in kwargs and kwargs['begin_indicator']:
                noise = feature_process(x.data, **kwargs)
                x.data += noise
        ################ END ####################
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor, **kwargs) -> Tensor:
         return self._forward_impl(x, **kwargs)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)