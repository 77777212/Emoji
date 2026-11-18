"""训练流程：train/val 循环、优化器与调度器、早停、checkpoint 与实验记录。

设计约定：
- 验证集使用 ``data.val_split``（FER2013 官方惯例为 PublicTest），
  ``PrivateTest`` 只在最终评估阶段使用，避免模型选择造成测试集泄漏；
- 每个 epoch 写一行 ``history.csv``，训练结束写 ``metrics.json`` 与 resolved config；
- ``best.pt`` 只保存权重与元信息，``last.pt`` 额外保存 optimizer/scheduler 状态，可用于 resume。
"""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from emoji.config import Config, save_config
from emoji.data import build_data_bundle
from emoji.losses import build_loss
from emoji.metrics import classification_report
from emoji.models import build_model_from_config
from emoji.utils import (
    append_csv_row,
    count_params,
    create_run_dir,
    format_duration,
    get_device,
    load_checkpoint,
    save_json,
    seed_everything,
)


@dataclass
class EpochStats:
    """单个 epoch 的训练/验证统计。"""

    loss: float
    acc: float
    macro_f1: float
    seconds: float
    n_samples: int
    lr: float


def calibrate_batch_norm(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> None:
    """用训练集前 ``max_batches`` 个 batch 校准 BN running stats（无梯度）。

    动机：torchvision MobileNetV3 的 BN momentum 默认是 0.01，小数据 + 1~2 epoch
    时 running stats 远未收敛，``model.eval()`` 会使用偏差很大的统计量，导致验证指标
    异常（例如始终预测同一类）。校准相当于"BN warmup"，不改变模型参数。
    """
    if max_batches <= 0:
        return
    batch_norms = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
    if not batch_norms:
        return
    saved_momentum = [bn.momentum for bn in batch_norms]
    for bn in batch_norms:
        bn.reset_running_stats()
        bn.momentum = None  # 使用累计移动平均
    was_training = model.training
    model.train()
    with torch.no_grad():
        for index, (images, _targets) in enumerate(loader):
            if index >= max_batches:
                break
            model(images.to(device))
    for bn, momentum in zip(batch_norms, saved_momentum):
        bn.momentum = momentum
    model.train(was_training)


def _move_batch(
    images: torch.Tensor, targets: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        images.to(device, non_blocking=True),
        targets.to(device, non_blocking=True),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    *,
    grad_clip: float = 5.0,
    amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
) -> EpochStats:
    """训练一个 epoch，返回训练集上的 loss/acc/macro-F1 与耗时。"""
    model.train()
    start = time.perf_counter()
    total_loss = 0.0
    n_samples = 0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for images, targets in loader:
        images, targets = _move_batch(images, targets, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
            loss = criterion(logits, targets)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_size = int(targets.size(0))
        total_loss += float(loss.detach().item()) * batch_size
        n_samples += batch_size
        all_preds.append(logits.detach().argmax(dim=1).cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

    y_pred = np.concatenate(all_preds) if all_preds else np.zeros(0, dtype=np.int64)
    y_true = np.concatenate(all_targets) if all_targets else np.zeros(0, dtype=np.int64)
    report = classification_report(y_true, y_pred, num_classes=num_classes)
    return EpochStats(
        loss=total_loss / max(1, n_samples),
        acc=float(report["acc"]),
        macro_f1=float(report["macro_f1"]),
        seconds=time.perf_counter() - start,
        n_samples=n_samples,
        lr=float(optimizer.param_groups[0]["lr"]),
    )


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """收集整个数据集上的 logits 与标签（不做 shuffle 时顺序与数据集一致）。"""
    model.eval()
    logits_list: list[torch.Tensor] = []
    target_list: list[torch.Tensor] = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits_list.append(model(images).detach().cpu())
        target_list.append(targets.detach().cpu())
    if not logits_list:
        raise RuntimeError("DataLoader 为空，无法收集预测结果")
    return torch.cat(logits_list, dim=0), torch.cat(target_list, dim=0)


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    class_names: Sequence[str] | None = None,
    criterion: nn.Module | None = None,
) -> dict[str, Any]:
    """在给定 DataLoader 上评估，返回 acc/macro-F1/每类指标/混淆矩阵（可选 loss）。"""
    logits, targets = collect_predictions(model, loader, device)
    y_pred = logits.argmax(dim=1).numpy()
    y_true = targets.numpy()
    report = classification_report(y_true, y_pred, num_classes=num_classes, class_names=class_names)
    if criterion is not None:
        with torch.no_grad():
            loss = criterion(logits.to(device), targets.to(device))
        report["loss"] = float(loss.item())
    report["n_samples"] = int(y_true.size)
    return report


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """按配置构建优化器（adamw / sgd）。"""
    if cfg.train.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
    if cfg.train.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.train.lr,
            momentum=cfg.train.momentum,
            weight_decay=cfg.train.weight_decay,
            nesterov=True,
        )
    raise ValueError(f"未知优化器 {cfg.train.optimizer!r}")


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: Config
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """按 epoch 步进的 warmup + 余弦退火调度器；``scheduler=none`` 时返回 None。"""
    if cfg.train.scheduler == "none":
        return None
    total_epochs = max(1, int(cfg.train.epochs))
    warmup = max(0, int(cfg.train.warmup_epochs))
    min_ratio = float(cfg.train.min_lr_ratio)

    def lr_lambda(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return float(epoch + 1) / float(warmup)
        progress = (epoch - warmup) / max(1, total_epochs - warmup)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _checkpoint_payload(
    model: nn.Module,
    cfg: Config,
    epoch: int,
    metrics: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "config": cfg.to_dict(),
        "epoch": int(epoch),
        "metrics": metrics,
        "model_name": cfg.model.name,
        "class_names": list(cfg.data.class_names),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    return payload


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _metric_line(prefix: str, metrics: dict[str, Any], seconds: float) -> str:
    return (
        f"{prefix}: loss={metrics.get('loss', float('nan')):.4f} "
        f"acc={metrics['acc']:.4f} macro_f1={metrics['macro_f1']:.4f} ({seconds:.1f}s)"
    )


def run_training(cfg: Config, resume: str | None = None) -> Path:
    """执行完整训练流程，返回本次实验的输出目录。"""
    cfg.validate()
    seed_everything(cfg.seed, deterministic=cfg.train.deterministic)
    if cfg.train.torch_num_threads > 0:
        torch.set_num_threads(int(cfg.train.torch_num_threads))

    device = get_device(cfg.train.device)
    run_dir = create_run_dir(cfg.output_dir, cfg.run_name)
    save_config(cfg, run_dir / "config.yaml")
    (run_dir / "command.txt").write_text(" ".join(sys.argv), encoding="utf-8")
    history_path = run_dir / "history.csv"
    metrics_path = run_dir / "metrics.json"

    data = build_data_bundle(cfg)
    class_counts = data.class_counts.tolist()
    print(
        f"[data] train={data.train_size}({data.train_split}) "
        f"val={data.val_size}({data.val_split}) "
        f"class_counts={class_counts}",
        flush=True,
    )

    model = build_model_from_config(cfg).to(device)
    if cfg.model.bn_warmup_batches > 0:
        calibrate_batch_norm(model, data.train_loader, device, cfg.model.bn_warmup_batches)
        print(
            f"[model] BN warmup: 用 {cfg.model.bn_warmup_batches} 个 batch 校准 running stats",
            flush=True,
        )
    n_params = count_params(model)
    print(
        f"[model] {cfg.model.name} params={n_params / 1e6:.2f}M "
        f"use_cbam={cfg.model.use_cbam} pretrained={cfg.model.pretrained}",
        flush=True,
    )

    criterion = build_loss(cfg.train, class_counts)
    if hasattr(criterion, "to"):
        criterion = criterion.to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    amp_enabled = cfg.train.amp != "off" and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    print(
        f"[device] {device} amp={'on' if amp_enabled else 'off'} "
        f"loss={cfg.train.loss} class_weight={cfg.train.class_weight}",
        flush=True,
    )
    print(f"[run] {run_dir}", flush=True)

    start_epoch = 0
    best_score = -math.inf
    best_epoch = 0
    bad_epochs = 0
    best_metrics: dict[str, Any] = {}
    if resume:
        ckpt = load_checkpoint(resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_metrics = dict(ckpt.get("metrics", {}))
        best_score = float(
            -best_metrics.get("loss", math.inf)
            if cfg.train.early_stop_monitor == "val_loss"
            else best_metrics.get("macro_f1", -math.inf)
        )
        best_epoch = int(ckpt.get("epoch", 0))
        print(f"[resume] 从 {resume} 恢复，start_epoch={start_epoch}", flush=True)

    train_start = time.perf_counter()
    epochs_run = 0
    monitor = cfg.train.early_stop_monitor
    for epoch in range(start_epoch, cfg.train.epochs):
        train_stats = train_one_epoch(
            model,
            data.train_loader,
            criterion,
            optimizer,
            device,
            cfg.data.num_classes,
            grad_clip=cfg.train.grad_clip,
            amp=amp_enabled,
            scaler=scaler,
        )
        val_metrics = evaluate_loader(
            model,
            data.val_loader,
            device,
            cfg.data.num_classes,
            cfg.data.class_names,
            criterion=criterion,
        )
        if scheduler is not None:
            scheduler.step()
        epochs_run += 1

        score = -float(val_metrics["loss"]) if monitor == "val_loss" else float(
            val_metrics["macro_f1"]
        )
        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
            best_metrics = dict(val_metrics)
            _save_checkpoint(
                run_dir / "best.pt",
                _checkpoint_payload(model, cfg, epoch, val_metrics),
            )
        else:
            bad_epochs += 1

        _save_checkpoint(
            run_dir / "last.pt",
            _checkpoint_payload(model, cfg, epoch, val_metrics, optimizer, scheduler),
        )
        append_csv_row(
            history_path,
            {
                "epoch": epoch + 1,
                "lr": f"{train_stats.lr:.8f}",
                "train_loss": f"{train_stats.loss:.6f}",
                "train_acc": f"{train_stats.acc:.6f}",
                "train_macro_f1": f"{train_stats.macro_f1:.6f}",
                "val_loss": f"{val_metrics['loss']:.6f}",
                "val_acc": f"{val_metrics['acc']:.6f}",
                "val_macro_f1": f"{val_metrics['macro_f1']:.6f}",
                "epoch_seconds": f"{train_stats.seconds:.3f}",
            },
        )
        marker = " *" if improved else ""
        print(
            f"[epoch {epoch + 1}/{cfg.train.epochs}] lr={train_stats.lr:.2e} "
            f"train_loss={train_stats.loss:.4f} train_acc={train_stats.acc:.4f} "
            f"train_f1={train_stats.macro_f1:.4f} | {_metric_line('val', val_metrics, train_stats.seconds)}"
            f"{marker}",
            flush=True,
        )
        patience = int(cfg.train.early_stop_patience)
        if patience > 0 and bad_epochs >= patience:
            print(f"[early-stop] 连续 {bad_epochs} 个 epoch 未提升，提前结束训练", flush=True)
            break

    # 用最优权重重新评估验证集并落盘 metrics.json
    best_path = run_dir / "best.pt"
    if best_path.is_file():
        best_ckpt = load_checkpoint(best_path, map_location="cpu")
        model.load_state_dict(best_ckpt["model"])
        best_metrics = dict(best_ckpt.get("metrics", best_metrics))
        best_epoch = int(best_ckpt.get("epoch", best_epoch))
    final_val = evaluate_loader(
        model,
        data.val_loader,
        device,
        cfg.data.num_classes,
        cfg.data.class_names,
        criterion=criterion,
    )
    duration = time.perf_counter() - train_start
    summary = {
        "run_dir": str(run_dir),
        "seed": cfg.seed,
        "device": str(device),
        "amp": amp_enabled,
        "model": {
            "name": cfg.model.name,
            "params": n_params,
            "params_m": round(n_params / 1e6, 4),
            "use_cbam": cfg.model.use_cbam,
            "pretrained": cfg.model.pretrained,
        },
        "dataset": {
            "csv_path": cfg.data.csv_path,
            "train_split": data.train_split,
            "val_split": data.val_split,
            "train_size": data.train_size,
            "val_size": data.val_size,
            "class_counts": class_counts,
        },
        "train": {
            "loss": cfg.train.loss,
            "class_weight": cfg.train.class_weight,
            "optimizer": cfg.train.optimizer,
            "lr": cfg.train.lr,
            "scheduler": cfg.train.scheduler,
            "epochs_configured": cfg.train.epochs,
            "epochs_run": epochs_run,
            "early_stop_patience": cfg.train.early_stop_patience,
        },
        "best_epoch": int(best_epoch) + 1,
        "best_val": best_metrics,
        "final_val": final_val,
        "duration_sec": round(duration, 3),
        "history_csv": str(history_path),
        "checkpoints": {
            "best": str(best_path),
            "last": str(run_dir / "last.pt"),
        },
        "config": cfg.to_dict(),
    }
    save_json(summary, metrics_path)
    print(
        f"[done] best_epoch={int(best_epoch) + 1} "
        f"val_acc={final_val['acc']:.4f} val_macro_f1={final_val['macro_f1']:.4f} "
        f"耗时={format_duration(duration)}",
        flush=True,
    )
    print(f"[outputs] {run_dir}", flush=True)
    return run_dir
