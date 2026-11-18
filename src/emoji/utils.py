"""通用工具：随机种子、设备选择、参数量统计、目录与文件落盘等。"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch


def now_str(fmt: str = "%Y%m%d-%H%M%S") -> str:
    """当前时间字符串，用于 run 命名。"""
    return time.strftime(fmt)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """固定 Python / NumPy / PyTorch 随机种子，尽量保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # warn_only=True：个别 CUDA 算子没有确定性实现时只告警，不中断训练
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (RuntimeError, AttributeError):  # 老版本 torch 或无 CUDA 环境
            pass


def get_device(preferred: str = "auto") -> torch.device:
    """按配置选择设备；``auto`` 依次尝试 CUDA → MPS → CPU。"""
    preferred = preferred.lower()
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("配置要求 cuda，但当前环境不可用（torch.cuda.is_available()=False）")
        return torch.device("cuda")
    if preferred == "mps":
        if not _mps_available():
            raise RuntimeError("配置要求 mps，但当前环境不可用")
        return torch.device("mps")
    if preferred != "auto":
        raise ValueError(f"未知设备 {preferred!r}，可选 auto/cpu/cuda/mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def _mps_available() -> bool:
    backends = getattr(torch.backends, "mps", None)
    return bool(backends is not None and backends.is_available())


def count_params(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """统计模型参数量；``trainable_only=True`` 时只统计可训练参数。"""
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def create_run_dir(output_dir: str | Path, run_name: str | None = None) -> Path:
    """创建实验输出目录；同名时自动追加时间后缀，避免覆盖历史结果。"""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if run_name:
        base = root / run_name
        if base.exists():
            base = root / f"{run_name}-{now_str('%H%M%S')}"
    else:
        base = root / now_str()
    base.mkdir(parents=True, exist_ok=False)
    return base


def _json_default(obj: Any) -> Any:
    """json.dump 的兜底转换：numpy / Path / Tensor 等。"""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


def save_json(obj: Any, path: str | Path) -> Path:
    """把任意可序列化对象写成 UTF-8 JSON 文件。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return target


def load_json(path: str | Path) -> Any:
    """读取 JSON 文件。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv_rows(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """把一组等字段的字典写成 CSV（以第一行的键顺序作为表头）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("rows 不能为空")
    fieldnames = list(rows[0].keys())
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return target


def append_csv_row(path: str | Path, row: Mapping[str, Any]) -> Path:
    """追加一行 CSV；文件不存在时用 ``row`` 的键顺序写表头。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    file_exists = target.is_file()
    with target.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    return target


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """加载本项目的 checkpoint（含模型权重、resolved config、指标）。

    本项目 checkpoint 由自己生成（非外部不可信文件），因此显式关闭
    ``weights_only`` 限制，以支持嵌套的普通 Python 对象。
    """
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"不是本项目的 checkpoint（缺少 model 字段）: {ckpt_path}")
    return ckpt


def format_duration(seconds: float) -> str:
    """把秒数格式化为 ``1h 02m 03s`` / ``12.3s`` 形式。"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {sec:02d}s"
    return f"{minutes}m {sec:02d}s"
