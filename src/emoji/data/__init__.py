"""数据子包：数据集、变换与 DataLoader 构建。"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from emoji.config import Config
from emoji.data.fer2013 import FER2013Dataset, count_rows_by_usage, normalize_split
from emoji.data.synth import generate_synth_dataset
from emoji.data.transforms import EvalTransform, TrainTransform, build_transforms

__all__ = [
    "DataBundle",
    "EvalTransform",
    "FER2013Dataset",
    "TrainTransform",
    "build_data_bundle",
    "build_test_loader",
    "build_transforms",
    "count_rows_by_usage",
    "generate_synth_dataset",
    "make_dataset",
    "normalize_split",
]


@dataclass(frozen=True)
class DataBundle:
    """训练所需的全部数据对象与统计信息。"""

    train_loader: DataLoader
    val_loader: DataLoader
    train_size: int
    val_size: int
    class_counts: np.ndarray
    num_classes: int
    train_split: str
    val_split: str


def make_dataset(cfg: Config, split: str, train: bool = False) -> FER2013Dataset:
    """按配置构建指定划分的 Dataset；``train=True`` 且开启增强时使用训练变换。"""
    train_tf, eval_tf = build_transforms(cfg)
    use_augment = bool(train and cfg.data.augment)
    transform = train_tf if use_augment else eval_tf
    return FER2013Dataset(
        csv_path=cfg.data.csv_path,
        split=split,
        image_size=cfg.data.image_size,
        transform=transform,
    )


def _make_loader(
    dataset: FER2013Dataset,
    cfg: Config,
    *,
    shuffle: bool,
    batch_size: int,
) -> DataLoader:
    """构建 DataLoader；CPU 环境下自动关闭 pin_memory 以避免无用开销。"""
    num_workers = max(0, int(cfg.data.num_workers))
    seed = int(cfg.seed)

    def _worker_init(worker_id: int) -> None:
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=bool(cfg.data.pin_memory and torch.cuda.is_available()),
        drop_last=False,
        persistent_workers=num_workers > 0,
        worker_init_fn=_worker_init if num_workers > 0 else None,
    )


def build_data_bundle(cfg: Config) -> DataBundle:
    """构建训练/验证 DataLoader 与训练集类别分布。

    - 训练集固定使用 ``Training`` 划分；
    - 验证集使用 ``data.val_split``（FER2013 官方惯例为 ``PublicTest``）；
    - ``PrivateTest`` 只在最终评估/鲁棒性/Grad-CAM 中使用，避免模型选择泄漏。
    """
    train_split = "Training"
    train_set = make_dataset(cfg, train_split, train=True)
    val_set = make_dataset(cfg, cfg.data.val_split, train=False)
    train_loader = _make_loader(train_set, cfg, shuffle=True, batch_size=cfg.data.batch_size)
    val_loader = _make_loader(
        val_set,
        cfg,
        shuffle=False,
        batch_size=max(1, int(cfg.eval.batch_size)),
    )
    class_counts = train_set.class_counts(cfg.data.num_classes)
    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        train_size=len(train_set),
        val_size=len(val_set),
        class_counts=class_counts,
        num_classes=cfg.data.num_classes,
        train_split=train_split,
        val_split=cfg.data.val_split,
    )


def build_test_loader(cfg: Config, split: str | None = None) -> tuple[DataLoader, FER2013Dataset]:
    """构建评估用 DataLoader（确定性变换、不打乱），返回 (loader, dataset)。"""
    split_name = split or cfg.eval.split
    dataset = make_dataset(cfg, split_name, train=False)
    loader = _make_loader(
        dataset,
        cfg,
        shuffle=False,
        batch_size=max(1, int(cfg.eval.batch_size)),
    )
    return loader, dataset
