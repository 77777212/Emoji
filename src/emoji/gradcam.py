"""Grad-CAM 可视化：查看模型在做表情分类时关注面部哪些区域。

用于两类分析：
- **错误分析**：错分样本的 CAM 是否落在背景/遮挡/非表情区域；
- **可解释性**：验证 CBAM 是否让注意力更集中在眼睛、嘴角等关键区域。

默认对预测类别生成 CAM（解释"模型为什么判成该类"），图中同时标注真实标签。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from emoji.config import Config, apply_overrides
from emoji.data import build_test_loader
from emoji.evaluate import load_model_from_checkpoint
from emoji.utils import get_device


class GradCAM:
    """标准 Grad-CAM：对目标层的激活按梯度均值加权求和，再 ReLU + 归一化。"""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: Tensor | None = None
        self.gradients: Tensor | None = None
        self._forward_handle = target_layer.register_forward_hook(self._save_activation)
        self._backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
        self.activations = output.detach()

    def _save_gradient(
        self,
        _module: nn.Module,
        _grad_inputs: tuple[Tensor, ...],
        grad_outputs: tuple[Tensor, ...],
    ) -> None:
        self.gradients = grad_outputs[0].detach()

    def generate(self, image: Tensor, class_idx: int | None = None) -> tuple[np.ndarray, int, float]:
        """对单张 ``(C,H,W)`` 图像生成 CAM。

        Returns:
            ``(cam, pred_idx, pred_prob)``，``cam`` 形状为 ``(H, W)``、范围 [0, 1]。
        """
        was_training = self.model.training
        self.model.eval()
        self.activations = None
        self.gradients = None
        batched = image.unsqueeze(0)
        with torch.enable_grad():
            logits = self.model(batched)
            probs = torch.softmax(logits, dim=1)
            pred_idx = int(logits.argmax(dim=1).item())
            target = pred_idx if class_idx is None else int(class_idx)
            self.model.zero_grad(set_to_none=True)
            logits[0, target].backward()
        self.model.train(was_training)
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM 未获取到激活或梯度，请检查目标层是否为卷积层")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))  # (1, 1, h, w)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-12)
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return cam[0, 0].cpu().numpy(), pred_idx, float(probs[0, pred_idx].item())

    def remove(self) -> None:
        """移除 hook（避免在后续前向中重复累积）。"""
        self._forward_handle.remove()
        self._backward_handle.remove()


def resolve_target_layer(model: nn.Module, layer_name: str = "") -> nn.Module:
    """解析 Grad-CAM 目标层：优先使用用户指定的点路径，否则用模型自带约定。"""
    if layer_name:
        module: nn.Module | None = model
        for part in layer_name.split("."):
            if module is None:
                break
            module = module[int(part)] if part.isdigit() else getattr(module, part, None)
        if not isinstance(module, nn.Module):
            raise ValueError(f"无法解析目标层 {layer_name!r}")
        return module
    getter = getattr(model, "gradcam_target_layer", None)
    if callable(getter):
        return getter()
    raise ValueError("模型未提供 gradcam_target_layer()，请用 --target-layer 指定目标层")


def _denormalize(
    image: Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    """把标准化后的 ``(C,H,W)`` 张量还原为 [0,1] 灰度图（多通道取均值）。"""
    mean_t = torch.as_tensor(list(mean), dtype=torch.float32).view(-1, 1, 1)
    std_t = torch.as_tensor(list(std), dtype=torch.float32).view(-1, 1, 1)
    if mean_t.numel() == 1:
        mean_t = mean_t.expand(image.shape[0], 1, 1)
    if std_t.numel() == 1:
        std_t = std_t.expand(image.shape[0], 1, 1)
    restored = (image.detach().cpu() * std_t + mean_t).clamp(0.0, 1.0)
    return restored.mean(dim=0).numpy()


def _save_cam_figure(
    gray: np.ndarray,
    cam: np.ndarray,
    title: str,
    path: Path,
) -> None:
    """保存单张「原图 + CAM 热力图」叠加图。"""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(3.2, 3.2), dpi=120)
    axis.imshow(gray, cmap="gray", vmin=0.0, vmax=1.0)
    axis.imshow(cam, cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
    axis.axis("off")
    axis.set_title(title, fontsize=7)
    fig.tight_layout(pad=0.2)
    fig.savefig(path)
    plt.close(fig)


def _save_cam_grid(
    items: list[tuple[np.ndarray, np.ndarray, str]],
    path: Path,
    columns: int = 4,
) -> None:
    """把多张 CAM 叠加图拼成一张网格图。"""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if not items:
        return
    rows = (len(items) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.0 * columns, 3.0 * rows), dpi=110)
    axes_array = np.atleast_1d(axes).ravel()
    for axis in axes_array:
        axis.axis("off")
    for axis, (gray, cam, title) in zip(axes_array, items):
        axis.imshow(gray, cmap="gray", vmin=0.0, vmax=1.0)
        axis.imshow(cam, cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
        axis.set_title(title, fontsize=7)
    fig.tight_layout(pad=0.4)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_gradcam_samples(
    ckpt_path: str | Path,
    split: str | None = None,
    out_dir: str | Path | None = None,
    top_k: int = 8,
    overrides: Sequence[str] = (),
    target_layer: str | None = None,
) -> Path:
    """对错误/正确样本各取 ``top_k`` 张生成 Grad-CAM 图，返回输出目录。"""
    if top_k <= 0:
        raise ValueError("top_k 必须为正整数")
    ckpt_file = Path(ckpt_path)
    model, _, ckpt = load_model_from_checkpoint(ckpt_file)
    cfg_dict = apply_overrides(ckpt["config"], overrides)
    cfg = Config.from_dict(cfg_dict)
    cfg.validate()
    split_name = split or cfg.eval.split
    device = get_device(cfg.eval.device)
    model.to(device).eval()
    loader, _dataset = build_test_loader(cfg, split_name)
    layer_name = target_layer if target_layer is not None else cfg.eval.target_layer
    layer = resolve_target_layer(model, layer_name)
    cam_engine = GradCAM(model, layer)

    try:
        images_list: list[Tensor] = []
        labels_list: list[Tensor] = []
        preds_list: list[Tensor] = []
        with torch.no_grad():
            for images, targets in loader:
                logits = model(images.to(device))
                images_list.append(images)
                labels_list.append(targets)
                preds_list.append(logits.argmax(dim=1).cpu())
        if not images_list:
            raise RuntimeError("测试集为空，无法生成 Grad-CAM")
        images = torch.cat(images_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        preds = torch.cat(preds_list, dim=0)

        wrong_indices = (preds != labels).nonzero().flatten().tolist()
        correct_indices = (preds == labels).nonzero().flatten().tolist()
        selected_wrong = wrong_indices[:top_k]
        selected_correct = correct_indices[:top_k]

        target_dir = (
            Path(out_dir) if out_dir is not None else ckpt_file.parent / f"gradcam_{split_name}"
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        class_names = list(cfg.data.class_names)
        grid_items: list[tuple[np.ndarray, np.ndarray, str]] = []
        for tag, indices in (("wrong", selected_wrong), ("correct", selected_correct)):
            for rank, index in enumerate(indices):
                image = images[index].to(device)
                cam, pred_idx, pred_prob = cam_engine.generate(image)
                gray = _denormalize(images[index], cfg.data.mean, cfg.data.std)
                true_idx = int(labels[index].item())
                title = (
                    f"true={class_names[true_idx]} pred={class_names[pred_idx]} "
                    f"p={pred_prob:.2f}"
                )
                filename = f"{tag}_{rank + 1}_true{true_idx}_pred{pred_idx}.png"
                _save_cam_figure(gray, cam, title, target_dir / filename)
                grid_items.append((gray, cam, f"{tag.upper()} #{rank + 1}: {title}"))
        _save_cam_grid(grid_items, target_dir / "gradcam_grid.png")
    finally:
        cam_engine.remove()

    print(
        f"[gradcam] split={split_name} wrong={len(selected_wrong)} correct={len(selected_correct)} "
        f"target_layer={layer_name or 'auto'}",
        flush=True,
    )
    print(f"[outputs] {target_dir}", flush=True)
    return target_dir
