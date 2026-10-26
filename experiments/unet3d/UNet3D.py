#!/usr/bin/env python3

"""A compact 3D U-Net for the fixed-depth SEGTHOR comparisons."""

import torch
from torch import nn, Tensor


def conv_block(input_channels: int, output_channels: int) -> nn.Sequential:
    layers = []
    for channels in (input_channels, output_channels):
        layers.extend([
            nn.Conv3d(channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(output_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        ])
    return nn.Sequential(*layers)


class UNet3D(nn.Module):
    def __init__(self, input_channels=1, num_classes=5,
                 widths=(8, 16, 32), pooling="isotropic"):
        super().__init__()
        if len(widths) != 3 or any(width <= 0 for width in widths):
            raise ValueError("3D U-Net needs three positive channel widths")
        if input_channels <= 0 or num_classes <= 0:
            raise ValueError("Input and output channel counts must be positive")
        if pooling not in ("isotropic", "inplane-first"):
            raise ValueError(f"Unknown pooling configuration: {pooling}")

        self.input_channels = input_channels
        self.widths = tuple(widths)
        self.pooling = pooling
        first_stride = (2, 2, 2) if pooling == "isotropic" else (1, 2, 2)
        self.spatial_divisors = tuple(2 * stride for stride in first_stride)
        first, second, third = self.widths

        self.encoder1 = conv_block(input_channels, first)
        self.pool1 = nn.MaxPool3d(first_stride)
        self.encoder2 = conv_block(first, second)
        self.pool2 = nn.MaxPool3d(2)
        self.bottleneck = conv_block(second, third)
        self.up2 = nn.ConvTranspose3d(third, second, kernel_size=2, stride=2)
        self.decoder2 = conv_block(second * 2, second)
        self.up1 = nn.ConvTranspose3d(second, first, kernel_size=first_stride,
                                      stride=first_stride)
        self.decoder1 = conv_block(first * 2, first)
        self.output = nn.Conv3d(first, num_classes, kernel_size=1)

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(module.weight, a=0.01,
                                        nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.InstanceNorm3d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 5 or images.shape[1] != self.input_channels:
            raise ValueError(f"Expected [B,{self.input_channels},D,H,W], got {images.shape}")
        if any(size % divisor for size, divisor in
               zip(images.shape[2:], self.spatial_divisors)):
            raise ValueError(f"Spatial shape must be divisible by {self.spatial_divisors}")

        skip1 = self.encoder1(images)
        skip2 = self.encoder2(self.pool1(skip1))
        features = self.bottleneck(self.pool2(skip2))
        features = self.decoder2(torch.cat((self.up2(features), skip2), dim=1))
        features = self.decoder1(torch.cat((self.up1(features), skip1), dim=1))
        return self.output(features)
