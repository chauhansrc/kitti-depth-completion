"""Depth completion network from Ma, Cavalheiro & Karaman,
"Self-Supervised Sparse-to-Dense: Self-Supervised Depth Completion from LiDAR and Monocular Camera" (ICRA 2019).

This is the supervised variant. RGB and sparse depth each get their own first conv
(late-ish fusion at the stem), then a shared ImageNet-pretrained ResNet encoder and a
transposed-conv decoder with skip connections back to every encoder stage.

`modality` selects the inputs so the fusion benefit can be measured:
    "rgbd" - camera + LiDAR (the multimodal model)
    "rgb"  - camera only (monocular depth baseline)
    "d"    - LiDAR only (sparse depth interpolation baseline)
"""

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34

MODALITIES = ("rgbd", "rgb", "d")


def conv_bn_relu(in_ch, out_ch, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


def convt_bn_relu(in_ch, out_ch, kernel_size, stride=1, padding=0, output_padding=0):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, output_padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


class DepthCompletionNet(nn.Module):
    def __init__(self, modality="rgbd", layers=34, pretrained=True, min_depth=0.9, output_scale=100.0):
        super().__init__()
        if modality not in MODALITIES:
            raise ValueError(f"modality must be one of {MODALITIES}")
        if layers not in (18, 34):
            raise ValueError("layers must be 18 or 34")
        self.modality = modality
        self.use_rgb = "rgb" in modality
        self.use_d = "d" in modality
        self.min_depth = min_depth
        # The paper's code scales the raw output by 100 so the last layer works in a small range.
        self.output_scale = output_scale

        # Stem: split 64 channels across inputs (rgbd -> 48 for RGB, 16 for depth).
        n = len(modality)
        if self.use_rgb:
            self.conv1_img = conv_bn_relu(3, 64 * 3 // n, 3, 1, 1)
        if self.use_d:
            self.conv1_d = conv_bn_relu(1, 64 // n, 3, 1, 1)

        # Encoder: ResNet stages without the original stem/max-pool, so stage 1 runs at full resolution.
        if layers == 18:
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        self.conv2 = backbone.layer1  # 64,  H
        self.conv3 = backbone.layer2  # 128, H/2
        self.conv4 = backbone.layer3  # 256, H/4
        self.conv5 = backbone.layer4  # 512, H/8
        del backbone
        self.conv6 = conv_bn_relu(512, 512, 3, 2, 1)  # 512, H/16

        # Decoder: each stage upsamples x2 and concatenates the matching encoder feature map.
        self.convt5 = convt_bn_relu(512, 256, 3, 2, 1, 1)        # -> H/8
        self.convt4 = convt_bn_relu(256 + 512, 128, 3, 2, 1, 1)  # -> H/4
        self.convt3 = convt_bn_relu(128 + 256, 64, 3, 2, 1, 1)   # -> H/2
        self.convt2 = convt_bn_relu(64 + 128, 64, 3, 2, 1, 1)    # -> H
        self.convt1 = convt_bn_relu(64 + 64, 64, 3, 1, 1)        # -> H
        self.convtf = nn.Conv2d(64 + 64, 1, 1)

    def forward(self, rgb, sparse):
        """rgb: Bx3xHxW (ImageNet-normalized), sparse: Bx1xHxW in meters. H and W must divide 16."""
        stems = []
        if self.use_rgb:
            stems.append(self.conv1_img(rgb))
        if self.use_d:
            stems.append(self.conv1_d(sparse))
        conv1 = torch.cat(stems, dim=1) if len(stems) > 1 else stems[0]

        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)

        y = self.convt5(conv6)
        y = self.convt4(torch.cat([y, conv5], dim=1))
        y = self.convt3(torch.cat([y, conv4], dim=1))
        y = self.convt2(torch.cat([y, conv3], dim=1))
        y = self.convt1(torch.cat([y, conv2], dim=1))
        y = self.convtf(torch.cat([y, conv1], dim=1))

        depth = self.output_scale * y
        if not self.training:
            # Valid depths in KITTI start around 1 m; clamping also keeps inverse-depth metrics finite.
            depth = torch.clamp(depth, min=self.min_depth)
        return depth
