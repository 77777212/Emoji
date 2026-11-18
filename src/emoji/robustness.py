"""鲁棒性实验：在测试集上施加高斯噪声 / 随机擦除 / 降分辨率（severity 1–5）。

扰动直接作用在**标准化之后**的张量上（此时干净图像的均值≈0、方差≈1），因此：
- 高斯噪声的 sigma 是相对标准差；
- 随机擦除用 0（即数据均值）填充被遮挡区域；
- 降分辨率用双线性下采样再上采样，模拟模糊/低清摄像头。

输出 ``robustness.csv``，包含每个扰动/强度下的 acc、macro_f1 及相对干净基线的
绝对下降（``acc_drop`` / ``macro_f1_drop``，单位为百分点）。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from emoji.config import Config, apply_overrides
from emoji.data import build_test_loader
from emoji.evaluate import load_model_from_checkpoint
from emoji.metrics import classification_report
from emoji.utils import get_device, write_csv_rows

# 各 severity 下高斯噪声的标准差（相对标准化后的图像）
_NOISE_SIGMA: dict[int, float] = {1: 0.05, 2: 0.10, 3: 0.15, 4: 0.22, 5: 0.30}
# 各 severity 下降分辨率的保留比例
_DOWNSCALE_RATIO: dict[int, float] = {1: 0.50, 2: 0.35, 3: 0.25, 4: 0.18, 5: 0.12}
# 随机擦除：每个 patch 的面积占全图比例
_ERASE_AREA_RATIO: dict[int, float] = {1: 0.02, 2: 0.03, 3: 0.05, 4: 0.07, 5: 0.10}

PERTURBATIONS: tuple[str, ...] = ("gaussian_noise", "random_erasing", "downscale")


def apply_gaussian_noise(
    images: Tensor, severity: int, generator: torch.Generator | None = None
) -> Tensor:
    """加性高斯噪声（在标准化空间）。"""
    sigma = _NOISE_SIGMA[severity]
    # 在 CPU 上生成噪声再搬运（CUDA Generator 与 CPU Generator 不通用，保证跨设备一致）
    noise = torch.randn(images.shape, generator=generator, dtype=images.dtype)
    return images + noise.to(images.device) * sigma


def apply_random_erasing(
    images: Tensor, severity: int, generator: torch.Generator | None = None
) -> Tensor:
    """随机擦除：每张图随机放置 ``severity`` 个遮挡块，用 0（数据均值）填充。"""
    _, _, height, width = images.shape
    area = height * width * _ERASE_AREA_RATIO[severity]
    side = max(2, round(area**0.5))
    side = min(side, height, width)
    patched = images.clone()
    for index in range(images.shape[0]):
        for _ in range(severity):
            top = torch.randint(0, height - side + 1, (1,), generator=generator).item()
            left = torch.randint(0, width - side + 1, (1,), generator=generator).item()
            patched[index, :, top : top + side, left : left + side] = 0.0
    return patched


def apply_downscale(
    images: Tensor, severity: int, generator: torch.Generator | None = None
) -> Tensor:
    """降分辨率：双线性下采样再上采样回原尺寸，模拟模糊/低清输入。"""
    _, _, height, width = images.shape
    ratio = _DOWNSCALE_RATIO[severity]
    target_h = max(4, round(height * ratio))
    target_w = max(4, round(width * ratio))
    small = F.interpolate(
        images, size=(target_h, target_w), mode="bilinear", align_corners=False, antialias=True
    )
    return F.interpolate(small, size=(height, width), mode="bilinear", align_corners=False)


def apply_perturbation(
    images: Tensor,
    name: str,
    severity: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    """按名称施加扰动；serverity 必须在 1..5。"""
    if severity not in (1, 2, 3, 4, 5):
        raise ValueError(f"severity 必须是 1..5，当前为 {severity}")
    if name == "gaussian_noise":
        return apply_gaussian_noise(images, severity, generator)
    if name == "random_erasing":
        return apply_random_erasing(images, severity, generator)
    if name == "downscale":
        return apply_downscale(images, severity, generator)
    raise ValueError(f"未知扰动 {name!r}；可选: {PERTURBATIONS}")


@torch.no_grad()
def _evaluate_perturbed(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    class_names: Sequence[str],
    perturbation: str,
    severity: int,
    seed: int,
    max_samples: int = 0,
) -> dict[str, Any]:
    """在施加扰动后的数据上评估。"""
    model.eval()
    # 用扰动名字符串的稳定哈希（sum(ord)），保证跨进程种子一致
    name_seed = sum(ord(char) for char in perturbation)
    generator = torch.Generator().manual_seed(seed + severity * 1000 + name_seed)
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    seen = 0
    for images, batch_targets in loader:
        images = images.to(device, non_blocking=True)
        if perturbation != "clean":
            images = apply_perturbation(images, perturbation, severity, generator)
        logits = model(images)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        targets.append(batch_targets.numpy())
        seen += int(batch_targets.size(0))
        if max_samples and seen >= max_samples:
            break
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets)
    if max_samples and y_true.size > max_samples:
        y_pred = y_pred[:max_samples]
        y_true = y_true[:max_samples]
    report = classification_report(y_true, y_pred, num_classes, class_names)
    return report


def run_robustness(
    ckpt_path: str | Path,
    split: str | None = None,
    out_dir: str | Path | None = None,
    severities: Sequence[int] | None = None,
    overrides: Sequence[str] = (),
    max_samples: int = 0,
) -> Path:
    """执行鲁棒性实验，写出 ``robustness.csv`` 并返回其路径。"""
    ckpt_file = Path(ckpt_path)
    model, _, ckpt = load_model_from_checkpoint(ckpt_file)
    # 允许用 --set 替换数据路径/评估参数；模型结构以 checkpoint 中的配置为准
    cfg_dict = apply_overrides(ckpt["config"], overrides)
    cfg = Config.from_dict(cfg_dict)
    cfg.validate()

    split_name = split or cfg.eval.split
    device = get_device(cfg.eval.device)
    model.to(device).eval()
    loader, _dataset = build_test_loader(cfg, split_name)
    severity_list = [int(item) for item in (severities or cfg.robustness.severities)]
    seed = int(cfg.robustness.seed)

    baseline = _evaluate_perturbed(
        model,
        loader,
        device,
        cfg.data.num_classes,
        cfg.data.class_names,
        perturbation="clean",
        severity=0,
        seed=seed,
        max_samples=max_samples,
    )
    rows: list[dict[str, Any]] = [
        {
            "perturbation": "clean",
            "severity": 0,
            "acc": round(float(baseline["acc"]), 6),
            "macro_f1": round(float(baseline["macro_f1"]), 6),
            "acc_drop": 0.0,
            "macro_f1_drop": 0.0,
            "n_samples": int(baseline["n_samples"]),
        }
    ]
    print(
        f"[robust] clean: acc={baseline['acc']:.4f} macro_f1={baseline['macro_f1']:.4f} "
        f"n={baseline['n_samples']}",
        flush=True,
    )

    for perturbation in PERTURBATIONS:
        for severity in severity_list:
            report = _evaluate_perturbed(
                model,
                loader,
                device,
                cfg.data.num_classes,
                cfg.data.class_names,
                perturbation=perturbation,
                severity=severity,
                seed=seed,
                max_samples=max_samples,
            )
            acc_drop = (baseline["acc"] - report["acc"]) * 100.0
            f1_drop = (baseline["macro_f1"] - report["macro_f1"]) * 100.0
            rows.append(
                {
                    "perturbation": perturbation,
                    "severity": severity,
                    "acc": round(float(report["acc"]), 6),
                    "macro_f1": round(float(report["macro_f1"]), 6),
                    "acc_drop": round(float(acc_drop), 3),
                    "macro_f1_drop": round(float(f1_drop), 3),
                    "n_samples": int(report["n_samples"]),
                }
            )
            print(
                f"[robust] {perturbation:<14s} sev={severity} "
                f"acc={report['acc']:.4f} (Δ{acc_drop:+.2f}pp) "
                f"f1={report['macro_f1']:.4f} (Δ{f1_drop:+.2f}pp)",
                flush=True,
            )

    target_dir = (
        Path(out_dir) if out_dir is not None else ckpt_file.parent / f"robustness_{split_name}"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    csv_path = target_dir / "robustness.csv"
    write_csv_rows(csv_path, rows)
    print(f"[outputs] {csv_path}", flush=True)
    return csv_path
