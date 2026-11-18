"""损失函数：Focal Loss、标签平滑交叉熵、类别加权交叉熵。

针对 FER 的两个常见问题：
- **类别不平衡**（FER2013 中 happy 约占 25%，disgust 不足 2%）：可用类别加权或 Focal Loss；
- **标签噪声**（众包标注存在歧义/错标）：可用 label smoothing 缓解过拟合。
注意：Focal Loss 会聚焦"难样本"，在标签噪声大时可能放大噪声，建议以消融实验验证，
不要默认它一定优于交叉熵。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from emoji.config import TrainConfig


def balanced_class_weights(class_counts: Sequence[int | float]) -> Tensor:
    """balanced 权重：``w_c = N / (C * n_c)``，并归一化到均值为 1。"""
    counts = torch.as_tensor(list(class_counts), dtype=torch.float32)
    if counts.numel() == 0 or torch.any(counts <= 0):
        raise ValueError(f"类别计数必须全为正数，且非空，当前为 {class_counts}")
    weights = counts.sum() / (counts.numel() * counts)
    return weights / weights.mean()


def effective_number_weights(
    class_counts: Sequence[int | float],
    beta: float = 0.999,
) -> Tensor:
    """Effective Number 权重（Cui et al., CVPR 2019），归一化到均值为 1。"""
    if not 0.0 < beta < 1.0:
        raise ValueError("beta 必须在 (0, 1) 内")
    counts = torch.as_tensor(list(class_counts), dtype=torch.float64)
    if counts.numel() == 0 or torch.any(counts <= 0):
        raise ValueError(f"类别计数必须全为正数，且非空，当前为 {class_counts}")
    effective_num = 1.0 - torch.pow(beta, counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.mean()
    return weights.to(torch.float32)


class FocalLoss(nn.Module):
    """Focal Loss（Lin et al., ICCV 2017）: ``-(1-p_t)^gamma * log(p_t)``。

    Args:
        gamma: 聚焦参数，0 时退化为普通交叉熵。
        weight: 可选的类别权重（等价于对每个样本的 loss 乘 alpha_t，
            与 CE 的 weight 语义一致，但归一化方式为权重和）。
        reduction: ``mean`` / ``sum`` / ``none``。

    说明：Focal Loss 默认不与 label smoothing 叠加（噪声标签场景建议用
    ``LabelSmoothingCE`` 或 ``ClassWeightedCE`` 做对照实验）。
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma 不能为负")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction 必须是 mean/sum/none")
        self.gamma = float(gamma)
        self.reduction = reduction
        if weight is not None:
            self.register_buffer("weight", weight.detach().clone().to(torch.float32))
        else:
            self.register_buffer("weight", None)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        nll = -log_probs.gather(1, targets.view(-1, 1)).squeeze(1)  # 每个样本的 -log(p_t)
        pt = torch.exp(-nll)
        loss = (1.0 - pt).pow(self.gamma) * nll
        if self.weight is not None:
            alpha = self.weight[targets]
            loss = loss * alpha
            if self.reduction == "mean":
                return loss.sum() / alpha.sum().clamp_min(1e-12)
            if self.reduction == "sum":
                return loss.sum()
            return loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class LabelSmoothingCE(nn.Module):
    """标签平滑交叉熵（PyTorch 内置实现的轻量封装）。"""

    def __init__(self, label_smoothing: float = 0.1, weight: Tensor | None = None) -> None:
        super().__init__()
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing 必须在 [0, 1) 内")
        self.label_smoothing = float(label_smoothing)
        self.criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=self.label_smoothing)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.criterion(logits, targets)


class ClassWeightedCE(nn.Module):
    """类别加权交叉熵；权重以 buffer 形式注册，可随 ``.to(device)`` 迁移。"""

    def __init__(self, class_weights: Tensor, label_smoothing: float = 0.0) -> None:
        super().__init__()
        if class_weights.ndim != 1:
            raise ValueError("class_weights 必须是一维张量")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing 必须在 [0, 1) 内")
        self.label_smoothing = float(label_smoothing)
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.detach().clone().to(torch.float32),
            label_smoothing=self.label_smoothing,
        )

    @property
    def class_weights(self) -> Tensor | None:
        """返回当前类别权重（便于记录到日志）。"""
        return self.criterion.weight

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.criterion(logits, targets)


def build_loss(cfg: TrainConfig, class_counts: Sequence[int | float] | None = None) -> nn.Module:
    """按 ``train.loss`` 构建损失函数。

    - ``ce``：交叉熵（可选类别权重）；
    - ``lsce``：标签平滑交叉熵（平滑系数取 ``train.label_smoothing``）；
    - ``focal``：Focal Loss（``train.focal_gamma``，默认不叠加标签平滑）；
    - ``cbce``：类别加权交叉熵，要求 ``train.class_weight`` 为 balanced/effective。
    """
    weight: Tensor | None = None
    if cfg.class_weight != "none":
        if class_counts is None:
            raise ValueError(f"train.class_weight={cfg.class_weight} 需要提供训练集类别计数")
        if cfg.class_weight == "balanced":
            weight = balanced_class_weights(class_counts)
        else:  # effective
            weight = effective_number_weights(class_counts, beta=cfg.effective_beta)

    if cfg.loss == "ce":
        return nn.CrossEntropyLoss(weight=weight)
    if cfg.loss == "lsce":
        return LabelSmoothingCE(label_smoothing=cfg.label_smoothing, weight=weight)
    if cfg.loss == "focal":
        return FocalLoss(gamma=cfg.focal_gamma, weight=weight)
    if cfg.loss == "cbce":
        if weight is None:
            raise ValueError(
                "loss=cbce 需要 train.class_weight ∈ {balanced, effective}，"
                f"当前为 {cfg.class_weight!r}"
            )
        return ClassWeightedCE(weight, label_smoothing=cfg.label_smoothing)
    raise ValueError(f"未知损失函数 {cfg.loss!r}，可选: ce/lsce/focal/cbce")
