"""CBAM（Convolutional Block Attention Module）注意力模块。

Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.
先做通道注意力（what），再做空间注意力（where），两路都是轻量级、不改变特征图尺寸。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ChannelAttention(nn.Module):
    """通道注意力：对每个通道分别做平均池化和最大池化，共享一个 MLP 后相加。"""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels 必须为正整数")
        hidden = max(1, channels // max(1, reduction))
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, _, _ = x.shape
        avg_feat = self.mlp(torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1))
        max_feat = self.mlp(torch.nn.functional.adaptive_max_pool2d(x, 1).flatten(1))
        weight = torch.sigmoid(avg_feat + max_feat).view(batch, channels, 1, 1)
        return x * weight


class SpatialAttention(nn.Module):
    """空间注意力：沿通道维取平均/最大值拼成 2 通道，再用 7×7 卷积得到空间权重图。"""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size 必须为奇数")
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)
        weight = torch.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * weight


class CBAM(nn.Module):
    """通道 + 空间注意力；输入输出尺寸完全一致，可直接插在任意 stage 后。"""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.spatial_attention(self.channel_attention(x))
