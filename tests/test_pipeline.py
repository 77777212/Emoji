"""端到端冒烟测试：训练 → 评估 → 鲁棒性 → Grad-CAM → 演示依赖。

所有测试均使用合成数据、CPU、1 个 epoch，不联网、不下载预训练权重。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from emoji.evaluate import evaluate_checkpoint
from emoji.gradcam import save_gradcam_samples
from emoji.robustness import run_robustness


def test_train_outputs(trained_run: Path) -> None:
    for name in ("best.pt", "last.pt", "history.csv", "metrics.json", "config.yaml"):
        assert (trained_run / name).is_file(), f"缺少训练产物 {name}"
    metrics = json.loads((trained_run / "metrics.json").read_text(encoding="utf-8"))
    assert 0.0 <= metrics["best_val"]["acc"] <= 1.0
    assert 0.0 <= metrics["best_val"]["macro_f1"] <= 1.0
    assert metrics["train"]["epochs_run"] == 1
    assert metrics["model"]["params"] > 0
    assert np.isfinite(metrics["final_val"]["acc"])
    # history.csv: 表头 + 1 个 epoch
    lines = (trained_run / "history.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert lines[0].split(",")[0] == "epoch"


def test_eval_outputs(trained_run: Path) -> None:
    artifacts = evaluate_checkpoint(
        trained_run / "best.pt",
        split="PrivateTest",
        out_dir=trained_run / "eval_private",
        measure_time=True,
        latency_iters=3,
    )
    assert artifacts.metrics_path.is_file()
    assert artifacts.per_class_path.is_file()
    assert artifacts.confusion_matrix_path.is_file()
    assert artifacts.confusion_matrix_path.stat().st_size > 1000
    metrics = json.loads(artifacts.metrics_path.read_text(encoding="utf-8"))
    assert metrics["split"] == "PrivateTest"
    assert 0.0 <= metrics["acc"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert metrics["latency"]["median_ms"] > 0
    rows = list(csv.DictReader(artifacts.per_class_path.open(encoding="utf-8")))
    assert len(rows) == 7
    assert {"label", "name", "precision", "recall", "f1", "support"} <= set(rows[0])


def test_robustness_outputs(trained_run: Path) -> None:
    csv_path = run_robustness(
        trained_run / "best.pt",
        split="PrivateTest",
        out_dir=trained_run / "robustness",
        severities=[1, 2],
    )
    assert csv_path.is_file()
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    # 1 行 clean + 3 种扰动 × 2 个 severity
    assert len(rows) == 1 + 3 * 2
    assert rows[0]["perturbation"] == "clean"
    assert float(rows[0]["acc_drop"]) == 0.0
    for row in rows[1:]:
        assert 0.0 <= float(row["acc"]) <= 1.0
        assert np.isfinite(float(row["acc_drop"]))
        assert row["perturbation"] in {"gaussian_noise", "random_erasing", "downscale"}


def test_gradcam_outputs(trained_run: Path) -> None:
    out_dir = save_gradcam_samples(
        trained_run / "best.pt",
        split="PrivateTest",
        out_dir=trained_run / "gradcam",
        top_k=2,
    )
    assert (out_dir / "gradcam_grid.png").is_file()
    samples = list(out_dir.glob("wrong_*.png")) + list(out_dir.glob("correct_*.png"))
    assert len(samples) >= 1
    assert (out_dir / "gradcam_grid.png").stat().st_size > 1000


def test_demo_module_is_lazy_and_fails_cleanly() -> None:
    """import 时不依赖 cv2；缺少 cv2 时给出带安装提示的清晰错误。"""
    assert "cv2" not in sys.modules
    import emoji.system.demo as demo_module

    assert "cv2" not in sys.modules
    try:
        import cv2  # noqa: F401

        pytest.skip("当前环境已安装 cv2，跳过缺失依赖分支")
    except ImportError:
        pass
    with pytest.raises(RuntimeError, match="opencv"):
        demo_module.run_demo("not-used.pt")


def test_crop_face_pure_numpy() -> None:
    from emoji.system.demo import crop_face

    frame = np.random.rand(100, 100, 3).astype(np.float32)
    face = crop_face(frame, (40, 40, 20, 20), margin=0.5)
    assert face.shape == (40, 40, 3)
    with pytest.raises(ValueError):
        crop_face(frame, (100, 100, 1, 1))


def test_cli_synth_data(tmp_path: Path) -> None:
    from emoji.cli import main

    out = tmp_path / "synth"
    code = main(
        ["synth-data", "--out", str(out), "--samples-per-class", "5", "--seed", "1"]
    )
    assert code == 0
    assert (out / "fer2013.csv").is_file()
    assert (out / "meta.json").is_file()


def test_cli_train_with_overrides(tmp_path: Path, synth_csv: Path) -> None:
    """验证 CLI 的 --set 覆盖链路（不依赖 configs/smoke.yaml）。"""
    from emoji.cli import main

    output_dir = tmp_path / "runs"
    code = main(
        [
            "train",
            "--set",
            f"data.csv_path={synth_csv}",
            "--set",
            "data.batch_size=8",
            "--set",
            "data.num_workers=0",
            "--set",
            "data.pin_memory=false",
            "--set",
            "model.name=mobilenet_v3_small_fer",
            "--set",
            "train.epochs=1",
            "--set",
            "train.warmup_epochs=0",
            "--set",
            "train.device=cpu",
            "--set",
            "train.torch_num_threads=1",
            "--set",
            f"output_dir={output_dir}",
            "--set",
            "run_name=cli-test",
        ]
    )
    assert code == 0
    run_dirs = list(output_dir.glob("cli-test*"))
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "best.pt").is_file()
