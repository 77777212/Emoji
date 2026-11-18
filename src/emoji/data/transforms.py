"""数据增强与归一化。

输入约定：数据集返回 ``(C, H, W)`` 的 float32 张量，像素范围 [0, 1]；
本模块负责训练增强、灰度转 3 通道与标准化。所有随机性来自 PyTorch 全局 RNG，
配合 :func:`emoji.utils.seed_everything` 即可复现。
"""

from __future__ import annotations

import torch
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from emoji.config import Config


def _channel_stats(
    mean: list[float] | tuple[float, ...],
    std: list[float] | tuple[float, ...],
    channels: int,
) -> tuple[Tensor, Tensor]:
    """把长度为 1 或 C 的均值/方差扩展成可广播的 (C, 1, 1) 张量。"""
    mean_t = torch.as_tensor(list(mean), dtype=torch.float32).view(-1, 1, 1)
    std_t = torch.as_tensor(list(std), dtype=torch.float32).view(-1, 1, 1)
    if mean_t.numel() == 1:
        mean_t = mean_t.expand(channels, 1, 1).contiguous()
    if std_t.numel() == 1:
        std_t = std_t.expand(channels, 1, 1).contiguous()
    if mean_t.numel() != channels or std_t.numel() != channels:
        raise ValueError(
            f"归一化参数长度（mean={mean_t.numel()}, std={std_t.numel()}）"
            f"与目标通道数 {channels} 不匹配"
        )
    return mean_t, std_t


def _to_out_channels(img: Tensor, out_chans: int) -> Tensor:
    """按需把灰度图复制成 3 通道（用于复用 ImageNet 预训练骨干）。"""
    if img.ndim != 3:
        raise ValueError(f"期望输入形状 (C, H, W)，实际为 {tuple(img.shape)}")
    in_chans = img.shape[0]
    if in_chans == out_chans:
        return img
    if in_chans == 1 and out_chans == 3:
        return img.repeat(3, 1, 1)
    raise ValueError(f"无法把 {in_chans} 通道输入转换为 {out_chans} 通道")


class TrainTransform:
    """训练数据增强：随机裁剪 → 水平翻转 → 小角度旋转 → 亮度/对比度扰动 → 标准化。

    增强幅度刻意保持温和：FER2013 本身分辨率低（48×48）且标签噪声大，
    过强的颜色/几何扰动容易进一步破坏表情线索。
    """

    def __init__(
        self,
        image_size: int = 48,
        out_chans: int = 1,
        mean: list[float] | tuple[float, ...] = (0.5,),
        std: list[float] | tuple[float, ...] = (0.5,),
        crop_padding: int = 4,
        hflip_p: float = 0.5,
        rotation_deg: float = 10.0,
        brightness: float = 0.2,
        contrast: float = 0.2,
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size 必须为正整数")
        self.image_size = image_size
        self.out_chans = out_chans
        self.mean, self.std = _channel_stats(mean, std, out_chans)
        self.crop_padding = max(0, int(crop_padding))
        self.hflip_p = float(hflip_p)
        self.rotation_deg = float(rotation_deg)
        self.brightness = float(brightness)
        self.contrast = float(contrast)

    def __call__(self, img: Tensor) -> Tensor:
        img = img.to(torch.float32)
        _, height, width = img.shape

        # 1) padding + 随机裁剪（保持输出尺寸不变）
        pad = min(self.crop_padding, height // 2, width // 2)
        if pad > 0:
            padded = TF.pad(img, [pad, pad, pad, pad], fill=0.0)
            top = int(torch.randint(0, 2 * pad + 1, (1,)).item())
            left = int(torch.randint(0, 2 * pad + 1, (1,)).item())
            img = padded[:, top : top + height, left : left + width]

        # 2) 水平翻转（表情左右对称，翻转不改变类别）
        if self.hflip_p > 0 and float(torch.rand(1).item()) < self.hflip_p:
            img = TF.hflip(img)

        # 3) 小角度旋转
        if self.rotation_deg > 0:
            angle = float(torch.empty(1).uniform_(-self.rotation_deg, self.rotation_deg).item())
            img = TF.rotate(
                img,
                angle,
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )

        # 4) 亮度 / 对比度扰动
        if self.brightness > 0:
            factor = float(torch.empty(1).uniform_(1.0 - self.brightness, 1.0 + self.brightness))
            img = TF.adjust_brightness(img, factor)
        if self.contrast > 0:
            factor = float(torch.empty(1).uniform_(1.0 - self.contrast, 1.0 + self.contrast))
            img = TF.adjust_contrast(img, factor)

        # 5) 灰度 → 3 通道 + 标准化
        img = _to_out_channels(img, self.out_chans)
        return (img - self.mean) / self.std


class EvalTransform:
    """确定性评估变换：仅做通道转换与标准化。"""

    def __init__(
        self,
        out_chans: int = 1,
        mean: list[float] | tuple[float, ...] = (0.5,),
        std: list[float] | tuple[float, ...] = (0.5,),
    ) -> None:
        self.out_chans = out_chans
        self.mean, self.std = _channel_stats(mean, std, out_chans)

    def __call__(self, img: Tensor) -> Tensor:
        img = _to_out_channels(img.to(torch.float32), self.out_chans)
        return (img - self.mean) / self.std


def build_transforms(cfg: Config) -> tuple[TrainTransform, EvalTransform]:
    """根据配置构建 (训练增强, 评估变换)。"""
    train_tf = TrainTransform(
        image_size=cfg.data.image_size,
        out_chans=cfg.data.model_in_chans,
        mean=cfg.data.mean,
        std=cfg.data.std,
    )
    eval_tf = EvalTransform(
        out_chans=cfg.data.model_in_chans,
        mean=cfg.data.mean,
        std=cfg.data.std,
    )
    return train_tf, eval_tf
