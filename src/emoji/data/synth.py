"""合成 FER 风格数据集（CSV 格式与 FER2013 完全一致）。

用途：在没有真实数据、也不联网的条件下跑通「生成数据 → 训练 → 评估 → 鲁棒性 →
Grad-CAM」全流程，并作为 pytest 的端到端冒烟数据。

每类表情使用一个固定基模式（不同位置的亮条）+ 高斯噪声 + 轻微平移，
类别间差异足够大，小模型 1~2 个 epoch 即可明显超过随机猜测（1/7 ≈ 14.3%）。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from emoji.config import DEFAULT_CLASS_NAMES

# 每条样本的噪声强度（相对 [0, 1] 灰度范围）
_NOISE_STD = 0.08


def _class_base_pattern(label: int, image_size: int) -> np.ndarray:
    """生成第 ``label`` 类的固定基模式，返回 (H, W) float32，范围 [0, 1]。"""
    # 背景亮度也随类别递增，给模型一个额外的易学线索（仅用于冒烟，不代表真实 FER）
    base = np.full((image_size, image_size), 0.05 + 0.05 * label, dtype=np.float32)
    bar = max(2, image_size // 10)
    if label < 4:
        # 类别 0~3：不同水平位置的竖条
        x = int((label + 1) * image_size / 5.5)
        base[:, x : x + bar] = 0.85
    else:
        # 类别 4~6：不同垂直位置的水平条
        y = int((label - 3) * image_size / 5.5)
        base[y : y + bar, :] = 0.85
    return base


def _split_counts(samples_per_class: int, train_frac: float, val_frac: float) -> tuple[int, int, int]:
    """计算每个类别在三个划分中的样本数（保证 val/test 至少各 1 条）。"""
    if samples_per_class < 3:
        raise ValueError("samples_per_class 至少为 3，才能让三个划分都有样本")
    if not 0.0 < train_frac < 1.0 or not 0.0 < val_frac < 1.0:
        raise ValueError("train_frac / val_frac 必须在 (0, 1) 内")
    n_train = max(1, round(samples_per_class * train_frac))
    n_val = max(1, round(samples_per_class * val_frac))
    if n_train + n_val > samples_per_class - 1:
        n_val = 1
        n_train = samples_per_class - 2
    n_test = samples_per_class - n_train - n_val
    return n_train, n_val, n_test


def generate_synth_dataset(
    out_dir: str | Path,
    samples_per_class: int = 32,
    seed: int = 0,
    image_size: int = 48,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Path:
    """生成合成 FER 数据集，返回写好的 ``fer2013.csv`` 路径。

    目录下会写两个文件：
    - ``fer2013.csv``：与 FER2013 同构（emotion,pixels,Usage）；
    - ``meta.json``：生成参数与各划分样本数，便于追溯。
    """
    if image_size <= 0:
        raise ValueError("image_size 必须为正整数")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_train, n_val, _ = _split_counts(samples_per_class, train_frac, val_frac)
    rng = np.random.default_rng(seed)
    num_classes = len(DEFAULT_CLASS_NAMES)

    rows: list[tuple[int, str, str]] = []
    counts = {split: 0 for split in ("Training", "PublicTest", "PrivateTest")}
    for label in range(num_classes):
        base = _class_base_pattern(label, image_size)
        for index in range(samples_per_class):
            noise = rng.normal(0.0, _NOISE_STD, size=base.shape).astype(np.float32)
            image = base + noise
            # 轻微随机平移 ±1 像素 + 全局亮度扰动，避免模型只记住绝对位置
            dy, dx = rng.integers(-1, 2, size=2)
            image = np.roll(image, (int(dy), int(dx)), axis=(0, 1))
            image = image * float(rng.uniform(0.92, 1.08)) + float(rng.uniform(-0.04, 0.04))
            image = np.clip(image, 0.0, 1.0)
            pixels = np.rint(image * 255.0).astype(np.uint8).ravel()
            if index < n_train:
                usage = "Training"
            elif index < n_train + n_val:
                usage = "PublicTest"
            else:
                usage = "PrivateTest"
            rows.append((label, " ".join(str(int(v)) for v in pixels), usage))
            counts[usage] += 1

    # 打乱行顺序，避免同类样本在训练时连续出现
    order = rng.permutation(len(rows))
    csv_path = out / "fer2013.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["emotion", "pixels", "Usage"])
        for idx in order:
            label, pixels, usage = rows[int(idx)]
            writer.writerow([label, pixels, usage])

    meta = {
        "generator": "emoji.data.synth",
        "seed": int(seed),
        "image_size": int(image_size),
        "samples_per_class": int(samples_per_class),
        "class_names": list(DEFAULT_CLASS_NAMES),
        "split_counts": counts,
        "noise_std": _NOISE_STD,
        "train_frac": float(train_frac),
        "val_frac": float(val_frac),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return csv_path
