"""评价指标：准确率、Macro-F1、每类 P/R/F1、混淆矩阵与推理时延统计。

实现基于 NumPy，不依赖 scikit-learn，便于在测试中直接校验数值。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def top1_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Top-1 准确率。"""
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """计算 ``num_classes × num_classes`` 混淆矩阵（行=真实类别，列=预测类别）。"""
    if num_classes <= 0:
        raise ValueError("num_classes 必须为正整数")
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true 与 y_pred 长度必须一致")
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        if not (0 <= true_label < num_classes) or not (0 <= pred_label < num_classes):
            raise ValueError(f"标签越界: true={true_label}, pred={pred_label}, C={num_classes}")
        matrix[true_label, pred_label] += 1
    return matrix


def per_class_metrics(matrix: np.ndarray) -> dict[str, np.ndarray]:
    """由混淆矩阵计算每类 precision / recall / f1 / support（除零时记 0）。"""
    matrix = np.asarray(matrix, dtype=np.float64)
    tp = np.diag(matrix)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2.0 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support.astype(np.int64),
    }


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """汇总指标，返回可直接 JSON 序列化的 dict。

    Returns:
        包含 ``acc``、``macro_f1``、``n_samples``、``per_class``（list[dict]）
        与 ``confusion_matrix``（list[list[int]]）的字典。
    """
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    per_class = per_class_metrics(matrix)
    names = list(class_names) if class_names is not None else [str(i) for i in range(num_classes)]
    if len(names) < num_classes:
        raise ValueError(f"class_names 长度 {len(names)} 小于 num_classes {num_classes}")
    macro_f1 = float(np.mean(per_class["f1"])) if num_classes > 0 else 0.0
    rows = [
        {
            "label": index,
            "name": names[index],
            "precision": float(per_class["precision"][index]),
            "recall": float(per_class["recall"][index]),
            "f1": float(per_class["f1"][index]),
            "support": int(per_class["support"][index]),
        }
        for index in range(num_classes)
    ]
    return {
        "acc": top1_accuracy(y_true, y_pred),
        "macro_f1": macro_f1,
        "n_samples": int(y_true.size),
        "per_class": rows,
        "confusion_matrix": matrix.tolist(),
    }


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: Sequence[str],
    path: str | Path,
    normalize: bool = False,
    title: str = "Confusion Matrix",
) -> Path:
    """把混淆矩阵画成 PNG（无 GUI 环境使用 Agg 后端）。"""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"混淆矩阵必须是方阵，当前形状 {matrix.shape}")
    if matrix.shape[0] != len(class_names):
        raise ValueError("混淆矩阵大小与 class_names 数量不一致")
    display = matrix.copy()
    if normalize:
        row_sum = display.sum(axis=1, keepdims=True)
        display = np.divide(display, row_sum, out=np.zeros_like(display), where=row_sum > 0)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 6.0), dpi=150)
    image = ax.imshow(display, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_xticklabels(list(class_names), rotation=45, ha="right")
    ax.set_yticks(ticks)
    ax.set_yticklabels(list(class_names))
    threshold = display.max() / 2.0 if display.size and display.max() > 0 else 0.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = f"{display[row, col]:.2f}" if normalize else f"{int(matrix[row, col])}"
            ax.text(
                col,
                row,
                value,
                ha="center",
                va="center",
                fontsize=7,
                color="white" if display[row, col] > threshold else "black",
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)
    return target


def measure_latency(
    model: nn.Module,
    input_size: int = 48,
    in_chans: int = 1,
    device: str | torch.device = "cpu",
    warmup: int = 3,
    iters: int = 10,
) -> dict[str, Any]:
    """测量 batch=1 的单张推理时延（毫秒）与 FPS。

    说明：报告时延应固定硬件与线程数；CUDA 上会调用 synchronize 保证计时准确。
    """
    if iters <= 0:
        raise ValueError("iters 必须为正整数")
    device_obj = torch.device(device)
    was_training = model.training
    model.eval()
    model.to(device_obj)
    dummy = torch.randn(1, in_chans, input_size, input_size, device=device_obj)
    timings: list[float] = []
    with torch.no_grad():
        for _ in range(max(0, warmup)):
            model(dummy)
            if device_obj.type == "cuda":
                torch.cuda.synchronize()
        for _ in range(iters):
            if device_obj.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(dummy)
            if device_obj.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)
    model.train(was_training)
    array = np.asarray(timings, dtype=np.float64)
    median_ms = float(np.median(array))
    return {
        "device": str(device_obj),
        "batch_size": 1,
        "input_size": int(input_size),
        "iters": int(iters),
        "median_ms": median_ms,
        "mean_ms": float(array.mean()),
        "p95_ms": float(np.percentile(array, 95)),
        "fps": float(1000.0 / median_ms) if median_ms > 0 else 0.0,
    }
