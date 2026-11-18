"""评估：加载 checkpoint 在指定数据划分上评估，并输出指标、每类报告与混淆矩阵。

产物（默认写到 checkpoint 所在目录下的 ``eval_<split>/``）：
- ``metrics.json``：整体指标、每类指标、混淆矩阵、时延统计、模型信息；
- ``per_class.csv``：每类 precision / recall / F1 / support；
- ``confusion_matrix.png``：混淆矩阵图。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from emoji.config import Config, apply_overrides
from emoji.data import build_test_loader
from emoji.metrics import measure_latency, plot_confusion_matrix
from emoji.models import build_model_from_config
from emoji.train import evaluate_loader
from emoji.utils import (
    count_params,
    get_device,
    load_checkpoint,
    save_json,
    write_csv_rows,
)


@dataclass
class EvalArtifacts:
    """一次评估产生的文件路径与指标。"""

    out_dir: Path
    metrics: dict[str, Any]
    metrics_path: Path
    per_class_path: Path
    confusion_matrix_path: Path


def load_model_from_checkpoint(
    ckpt_path: str | Path,
    device: str | torch.device | None = None,
) -> tuple[nn.Module, Config, dict[str, Any]]:
    """加载 checkpoint，重建模型（结构与 resolved config 完全一致）并切到 eval 模式。"""
    ckpt = load_checkpoint(ckpt_path)
    cfg = Config.from_dict(ckpt["config"])
    cfg.validate()
    model = build_model_from_config(cfg)
    try:
        model.load_state_dict(ckpt["model"])
    except RuntimeError as exc:
        raise RuntimeError(
            f"checkpoint 权重与配置不匹配（{cfg.model.name}）：{exc}\n"
            "请确认没有用修改过模型结构的 --set 覆盖项去加载权重。"
        ) from exc
    device_obj = get_device(cfg.eval.device if device is None else str(device))
    model.to(device_obj).eval()
    return model, cfg, ckpt


def evaluate_checkpoint(
    ckpt_path: str | Path,
    split: str | None = None,
    out_dir: str | Path | None = None,
    overrides: Sequence[str] = (),
    save_plot: bool = True,
    measure_time: bool = True,
    latency_iters: int = 10,
) -> EvalArtifacts:
    """在指定划分上评估 checkpoint。

    Args:
        ckpt_path: checkpoint 路径（``best.pt`` 或 ``last.pt``）。
        split: 数据划分，默认取 checkpoint 中解析出的 ``eval.split``。
        out_dir: 输出目录，默认 ``<checkpoint 目录>/eval_<split>``。
        overrides: ``--set`` 风格的配置覆盖（如更换 csv_path）。
        save_plot: 是否绘制混淆矩阵 PNG。
        measure_time: 是否统计 batch=1 推理时延。
    """
    ckpt_file = Path(ckpt_path)
    ckpt = load_checkpoint(ckpt_file)
    cfg_dict = apply_overrides(ckpt["config"], overrides)
    cfg = Config.from_dict(cfg_dict)
    cfg.validate()

    model = build_model_from_config(cfg)
    try:
        model.load_state_dict(ckpt["model"])
    except RuntimeError as exc:
        raise RuntimeError(
            f"checkpoint 权重与配置不匹配（{cfg.model.name}）：{exc}\n"
            "提示：--set 覆盖项只能改数据/评估相关字段，不应改变模型结构。"
        ) from exc

    split_name = split or cfg.eval.split
    device = get_device(cfg.eval.device)
    model.to(device).eval()
    loader, dataset = build_test_loader(cfg, split_name)

    report = evaluate_loader(
        model,
        loader,
        device,
        cfg.data.num_classes,
        cfg.data.class_names,
    )
    latency: dict[str, Any] | None = None
    if measure_time:
        latency = measure_latency(
            model,
            input_size=cfg.data.image_size,
            in_chans=cfg.data.model_in_chans,
            device=device,
            iters=latency_iters,
        )

    target_dir = (
        Path(out_dir) if out_dir is not None else ckpt_file.parent / f"eval_{split_name}"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {
        "checkpoint": str(ckpt_file),
        "split": split_name,
        "n_samples": len(dataset),
        "class_names": list(cfg.data.class_names),
        "model": {
            "name": cfg.model.name,
            "params": count_params(model),
            "use_cbam": cfg.model.use_cbam,
        },
        "acc": report["acc"],
        "macro_f1": report["macro_f1"],
        "per_class": report["per_class"],
        "confusion_matrix": report["confusion_matrix"],
        "latency": latency,
        "eval_config": cfg.to_dict()["eval"],
        "data": {
            "csv_path": cfg.data.csv_path,
            "image_size": cfg.data.image_size,
            "in_chans": cfg.data.model_in_chans,
        },
    }
    metrics_path = target_dir / "metrics.json"
    save_json(metrics, metrics_path)

    per_class_path = target_dir / "per_class.csv"
    write_csv_rows(per_class_path, report["per_class"])

    cm_path = target_dir / "confusion_matrix.png"
    if save_plot:
        plot_confusion_matrix(
            report["confusion_matrix"],
            cfg.data.class_names,
            cm_path,
            title=f"Confusion Matrix ({split_name}, n={len(dataset)})",
        )

    print(
        f"[eval] split={split_name} n={len(dataset)} acc={report['acc']:.4f} "
        f"macro_f1={report['macro_f1']:.4f}",
        flush=True,
    )
    if latency is not None:
        print(
            f"[latency] batch=1 median={latency['median_ms']:.2f}ms "
            f"p95={latency['p95_ms']:.2f}ms fps={latency['fps']:.1f}",
            flush=True,
        )
    print(f"[outputs] {target_dir}", flush=True)
    return EvalArtifacts(
        out_dir=target_dir,
        metrics=metrics,
        metrics_path=metrics_path,
        per_class_path=per_class_path,
        confusion_matrix_path=cm_path,
    )
