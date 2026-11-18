"""FER2013 CSV 数据集。

数据格式（Kaggle FER2013 原始文件 ``fer2013.csv``）::

    emotion,pixels,Usage
    0,"70 70 70 ...",Training
    ...

- ``emotion``：0..6 的整数标签，对应 angry/disgust/fear/happy/sad/surprise/neutral；
- ``pixels``：48×48=2304 个空格分隔的灰度像素值（0..255）；
- ``Usage``：``Training`` / ``PublicTest`` / ``PrivateTest`` 三种官方划分。

本 Dataset 返回 ``(1×H×W float32 张量, 类别标签)``，像素缩放到 [0, 1]；
``image_size`` 不等于 48 时用双线性插值缩放到目标尺寸（用于 96 输入消融）。
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# FER2013 原始图像边长
FER2013_NATIVE_SIZE = 48

# 兼容大小写与常见别名，统一映射到 CSV 中 Usage 的三种官方取值
_SPLIT_ALIASES: dict[str, str] = {
    "train": "Training",
    "training": "Training",
    "val": "PublicTest",
    "valid": "PublicTest",
    "validation": "PublicTest",
    "public": "PublicTest",
    "publictest": "PublicTest",
    "test": "PrivateTest",
    "private": "PrivateTest",
    "privatetest": "PrivateTest",
}


def normalize_split(split: str) -> str:
    """把用户输入的划分名归一化为 ``Training``/``PublicTest``/``PrivateTest``。"""
    key = str(split).strip().lower()
    if key in _SPLIT_ALIASES:
        return _SPLIT_ALIASES[key]
    raise ValueError(
        f"未知数据集划分 {split!r}；可选: Training / PublicTest / PrivateTest "
        "（也接受 train/val/test 等别名）"
    )


def _parse_pixels(pixel_str: str) -> np.ndarray:
    """把空格分隔的像素字符串解析成 (48, 48) uint8 数组。"""
    values = np.fromstring(pixel_str, sep=" ", dtype=np.float32)
    if values.size != FER2013_NATIVE_SIZE * FER2013_NATIVE_SIZE:
        raise ValueError(f"像素数量异常：期望 2304，实际 {values.size}")
    return values.reshape(FER2013_NATIVE_SIZE, FER2013_NATIVE_SIZE)


class FER2013Dataset(Dataset):
    """FER2013 CSV 数据集。

    Args:
        csv_path: ``fer2013.csv`` 路径。
        split: 划分名（Training/PublicTest/PrivateTest 或别名）。
        image_size: 输出边长；48 时直接使用原始图像，其它值用双线性插值缩放。
        transform: 作用于 ``(1, H, W)`` float 张量的可调用对象。
    """

    def __init__(
        self,
        csv_path: str | Path,
        split: str = "Training",
        image_size: int = FER2013_NATIVE_SIZE,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.split = normalize_split(split)
        self.image_size = int(image_size)
        self.transform = transform
        if self.image_size <= 0:
            raise ValueError("image_size 必须为正整数")
        if not self.csv_path.is_file():
            raise FileNotFoundError(
                f"未找到 FER2013 CSV: {self.csv_path}\n"
                "请按 data/README.md 从 Kaggle 下载 FER2013 并解压/放置到 "
                "data/raw/fer2013/fer2013.csv，或先运行合成数据冒烟命令：\n"
                "  emoji synth-data --out data/synth/smoke --samples-per-class 24 --seed 0"
            )

        # 只保存 (pixels 字符串, label)，像素在 __getitem__ 中惰性解析，内存占用小
        self._samples: list[tuple[str, int]] = []
        with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = {"emotion", "pixels", "Usage"} - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"FER2013 CSV 缺少必需列 {sorted(missing)}: {self.csv_path}")
            for row in reader:
                usage = str(row["Usage"]).strip()
                if usage != self.split:
                    continue
                self._samples.append((row["pixels"], int(row["emotion"])))

        if not self._samples:
            raise ValueError(
                f"{self.csv_path} 中没有 Usage={self.split} 的样本；"
                "请检查数据文件是否完整，或换用其它划分"
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        pixel_str, label = self._samples[index]
        image = _parse_pixels(pixel_str)
        if self.image_size != FER2013_NATIVE_SIZE:
            resized = Image.fromarray(image.astype(np.uint8)).resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            image = np.asarray(resized, dtype=np.float32)
        else:
            image = image.astype(np.float32)
        tensor = torch.from_numpy(image / 255.0).unsqueeze(0)  # (1, H, W)
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor, label

    # ------------------------------------------------------------------ #
    # 便捷统计
    # ------------------------------------------------------------------ #
    def labels(self) -> np.ndarray:
        """返回当前划分的全部标签数组。"""
        return np.asarray([label for _, label in self._samples], dtype=np.int64)

    def class_counts(self, num_classes: int = 7) -> np.ndarray:
        """返回每个类别的样本数。"""
        return np.bincount(self.labels(), minlength=num_classes)

    def __repr__(self) -> str:  # pragma: no cover - 仅调试展示
        return (
            f"FER2013Dataset(csv_path={str(self.csv_path)!r}, split={self.split!r}, "
            f"n={len(self)}, image_size={self.image_size})"
        )


def count_rows_by_usage(csv_path: str | Path) -> dict[str, int]:
    """统计 CSV 中各 Usage 划分的样本数（不解析像素，速度快）。"""
    counts: dict[str, int] = {}
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            usage = str(row["Usage"]).strip()
            counts[usage] = counts.get(usage, 0) + 1
    return counts
