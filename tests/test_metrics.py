"""评价指标测试：用小型已知数据校验 acc / macro-F1 / 混淆矩阵数值。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from emoji.metrics import (
    classification_report,
    confusion_matrix,
    measure_latency,
    per_class_metrics,
    plot_confusion_matrix,
    top1_accuracy,
)


def test_confusion_matrix_and_report_known_values() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    matrix = confusion_matrix(y_true, y_pred, num_classes=2)
    assert matrix.tolist() == [[1, 1], [0, 2]]

    report = classification_report(y_true, y_pred, num_classes=2, class_names=["a", "b"])
    assert report["acc"] == pytest.approx(0.75)
    assert report["n_samples"] == 4
    per_class = report["per_class"]
    assert per_class[0]["precision"] == pytest.approx(1.0)
    assert per_class[0]["recall"] == pytest.approx(0.5)
    assert per_class[0]["f1"] == pytest.approx(2.0 / 3.0)
    assert per_class[0]["support"] == 2
    assert per_class[1]["precision"] == pytest.approx(2.0 / 3.0)
    assert per_class[1]["recall"] == pytest.approx(1.0)
    assert report["macro_f1"] == pytest.approx(((2.0 / 3.0) + 0.8) / 2.0)


def test_perfect_predictions() -> None:
    y_true = np.arange(7)
    report = classification_report(y_true, y_true.copy(), num_classes=7)
    assert report["acc"] == pytest.approx(1.0)
    assert report["macro_f1"] == pytest.approx(1.0)


def test_top1_accuracy_empty() -> None:
    assert top1_accuracy(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)) == 0.0


def test_confusion_matrix_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="标签越界"):
        confusion_matrix(np.array([0, 3]), np.array([0, 1]), num_classes=3)


def test_per_class_metrics_with_missing_class() -> None:
    # 类别 1 从未出现：precision/recall/f1 记为 0，support 为 0
    matrix = np.array([[2, 0], [0, 0]])
    stats = per_class_metrics(matrix)
    assert stats["support"].tolist() == [2, 0]
    assert stats["f1"].tolist() == [1.0, 0.0]


def test_plot_confusion_matrix(tmp_path: Path) -> None:
    matrix = np.array([[5, 1, 0], [0, 4, 1], [2, 0, 3]])
    path = plot_confusion_matrix(matrix, ["a", "b", "c"], tmp_path / "cm.png")
    assert path.is_file()
    assert path.stat().st_size > 1000


def test_measure_latency_positive() -> None:
    model = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 7),
    )
    stats = measure_latency(model, input_size=16, in_chans=1, iters=3, warmup=1)
    assert stats["median_ms"] > 0
    assert stats["mean_ms"] > 0
    assert stats["fps"] > 0
    assert stats["batch_size"] == 1
    assert torch.isfinite(torch.tensor(stats["median_ms"]))
