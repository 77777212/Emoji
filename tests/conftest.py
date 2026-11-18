"""pytest 共享配置与 fixture。

全部测试使用 CPU + 合成数据，不下载任何真实数据集或预训练权重。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import torch

# 支持未安装为包时直接从 src 运行测试
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

torch.set_num_threads(1)
torch.manual_seed(0)


def build_test_config(synth_csv: Path, output_dir: Path, **overrides: Any) -> Any:
    """构建一个小而快的测试配置（MobileNetV3-small + 1 epoch + CPU）。"""
    from emoji.config import Config, deep_merge

    base: dict[str, Any] = {
        "seed": 0,
        "output_dir": str(output_dir),
        "run_name": "unit",
        "data": {
            "csv_path": str(synth_csv),
            "batch_size": 8,
            "num_workers": 0,
            "pin_memory": False,
            "image_size": 48,
        },
        "model": {"name": "mobilenet_v3_small_fer", "num_classes": 7},
        "train": {
            "epochs": 1,
            "lr": 1e-3,
            "warmup_epochs": 0,
            "early_stop_patience": 2,
            "torch_num_threads": 1,
            "device": "cpu",
        },
        "eval": {"batch_size": 16, "device": "cpu", "split": "PrivateTest"},
        "robustness": {"severities": [1, 2], "split": "PrivateTest"},
    }
    merged = deep_merge(base, overrides)
    cfg = Config.from_dict(merged)
    cfg.validate()
    return cfg


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """会话级合成数据集（每类 12 张：训练 8 / 验证 2 / 测试 2）。"""
    from emoji.data.synth import generate_synth_dataset

    out = tmp_path_factory.mktemp("synth-data")
    generate_synth_dataset(out, samples_per_class=12, seed=0, image_size=48)
    return out


@pytest.fixture(scope="session")
def synth_csv(synth_dir: Path) -> Path:
    """合成数据 CSV 路径。"""
    return synth_dir / "fer2013.csv"


@pytest.fixture(scope="session")
def trained_run(tmp_path_factory: pytest.TempPathFactory, synth_csv: Path) -> Path:
    """会话级：训练 1 个 epoch，供评估/鲁棒性/Grad-CAM 测试复用。"""
    from emoji.train import run_training

    out = tmp_path_factory.mktemp("runs")
    cfg = build_test_config(synth_csv, out)
    return run_training(cfg)
