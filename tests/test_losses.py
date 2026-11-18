"""损失函数测试：数值有限、形状正确、weight/label smoothing 生效。"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from emoji.config import TrainConfig
from emoji.losses import (
    ClassWeightedCE,
    FocalLoss,
    LabelSmoothingCE,
    balanced_class_weights,
    build_loss,
    effective_number_weights,
)


def _logits_and_targets() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    logits = torch.randn(8, 7, requires_grad=True)
    targets = torch.randint(0, 7, (8,))
    return logits, targets


def test_focal_gamma_zero_equals_cross_entropy() -> None:
    logits, targets = _logits_and_targets()
    focal = FocalLoss(gamma=0.0)
    ce = nn.CrossEntropyLoss()
    assert torch.allclose(focal(logits, targets), ce(logits, targets), atol=1e-6)


def test_focal_with_class_weights_is_finite_and_backward() -> None:
    logits, targets = _logits_and_targets()
    weights = balanced_class_weights([100, 20, 10, 5, 3, 2, 1])
    loss = FocalLoss(gamma=2.0, weight=weights)(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_balanced_class_weights_normalized() -> None:
    weights = balanced_class_weights([100, 10, 1])
    assert weights.shape == (3,)
    assert float(weights.mean()) == pytest.approx(1.0, abs=1e-5)
    assert weights[2] > weights[0]


def test_effective_number_weights() -> None:
    weights = effective_number_weights([100, 10, 1], beta=0.999)
    assert weights.shape == (3,)
    assert torch.isfinite(weights).all()
    assert float(weights.mean()) == pytest.approx(1.0, abs=1e-4)
    assert weights[2] > weights[0]


def test_class_weights_reject_zero_counts() -> None:
    with pytest.raises(ValueError):
        balanced_class_weights([10, 0, 1])
    with pytest.raises(ValueError):
        effective_number_weights([10, 0, 1])


def test_build_loss_variants() -> None:
    counts = [50, 20, 10, 10, 5, 3, 2]
    assert isinstance(build_loss(TrainConfig(loss="ce"), counts), nn.CrossEntropyLoss)
    assert isinstance(build_loss(TrainConfig(loss="lsce"), counts), LabelSmoothingCE)
    assert isinstance(build_loss(TrainConfig(loss="focal"), counts), FocalLoss)
    weighted = build_loss(TrainConfig(loss="cbce", class_weight="balanced"), counts)
    assert isinstance(weighted, ClassWeightedCE)
    assert weighted.class_weights is not None
    assert torch.isfinite(weighted.class_weights).all()


def test_build_loss_cbce_requires_class_weight() -> None:
    with pytest.raises(ValueError, match="class_weight"):
        build_loss(TrainConfig(loss="cbce", class_weight="none"), [10] * 7)


def test_build_loss_weighted_requires_counts() -> None:
    with pytest.raises(ValueError, match="类别计数"):
        build_loss(TrainConfig(loss="ce", class_weight="balanced"), None)


def test_label_smoothing_changes_loss_value() -> None:
    logits, targets = _logits_and_targets()
    plain = nn.CrossEntropyLoss()(logits, targets)
    smooth = LabelSmoothingCE(label_smoothing=0.2)(logits, targets)
    assert torch.isfinite(smooth)
    assert not torch.allclose(plain, smooth)


def test_loss_outputs_are_scalars() -> None:
    logits, targets = _logits_and_targets()
    counts = [50, 20, 10, 10, 5, 3, 2]
    for loss_name in ("ce", "lsce", "focal", "cbce"):
        criterion = build_loss(
            TrainConfig(loss=loss_name, class_weight="balanced"), counts
        )
        value = criterion(logits, targets)
        assert value.ndim == 0
        assert torch.isfinite(value)
