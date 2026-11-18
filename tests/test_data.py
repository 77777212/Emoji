"""数据集、CSV 解析、合成数据与数据增强测试。"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from emoji.data.fer2013 import FER2013Dataset, count_rows_by_usage, normalize_split
from emoji.data.synth import generate_synth_dataset
from emoji.data.transforms import EvalTransform, TrainTransform

PIXELS = " ".join(["128"] * 2304)


def _write_fer_csv(path: Path, rows: list[tuple[int, str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["emotion", "pixels", "Usage"])
        for emotion, usage, pixels in rows:
            writer.writerow([emotion, pixels, usage])
    return path


def test_split_aliases() -> None:
    assert normalize_split("train") == "Training"
    assert normalize_split("Training") == "Training"
    assert normalize_split("val") == "PublicTest"
    assert normalize_split("test") == "PrivateTest"
    assert normalize_split("PrivateTest") == "PrivateTest"
    with pytest.raises(ValueError):
        normalize_split("dev")


def test_dataset_split_and_shape(tmp_path: Path) -> None:
    csv_path = _write_fer_csv(
        tmp_path / "fer2013.csv",
        [
            (0, "Training", PIXELS),
            (3, "Training", PIXELS),
            (5, "PublicTest", PIXELS),
            (6, "PrivateTest", PIXELS),
        ],
    )
    train = FER2013Dataset(csv_path, split="Training")
    assert len(train) == 2
    image, label = train[0]
    assert image.shape == (1, 48, 48)
    assert image.dtype == torch.float32
    assert float(image.min()) >= 0.0 and float(image.max()) <= 1.0
    assert label in (0, 3)
    assert len(FER2013Dataset(csv_path, split="PublicTest")) == 1
    assert len(FER2013Dataset(csv_path, split="PrivateTest")) == 1
    assert train.class_counts(7).tolist() == [1, 0, 0, 1, 0, 0, 0]
    assert count_rows_by_usage(csv_path) == {
        "Training": 2,
        "PublicTest": 1,
        "PrivateTest": 1,
    }


def test_dataset_image_size_96(tmp_path: Path) -> None:
    csv_path = _write_fer_csv(tmp_path / "fer2013.csv", [(2, "Training", PIXELS)])
    dataset = FER2013Dataset(csv_path, split="Training", image_size=96)
    image, _ = dataset[0]
    assert image.shape == (1, 96, 96)


def test_missing_csv_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="FER2013"):
        FER2013Dataset(tmp_path / "missing.csv")


def test_bad_pixel_count_raises(tmp_path: Path) -> None:
    csv_path = _write_fer_csv(tmp_path / "fer2013.csv", [(0, "Training", "1 2 3")])
    dataset = FER2013Dataset(csv_path, split="Training")
    with pytest.raises(ValueError, match="像素数量"):
        _ = dataset[0]


def test_empty_split_raises(tmp_path: Path) -> None:
    csv_path = _write_fer_csv(tmp_path / "fer2013.csv", [(0, "Training", PIXELS)])
    with pytest.raises(ValueError, match="没有 Usage="):
        FER2013Dataset(csv_path, split="PrivateTest")


def test_synth_dataset_generation(tmp_path: Path) -> None:
    csv_path = generate_synth_dataset(tmp_path / "synth", samples_per_class=6, seed=1)
    counts = count_rows_by_usage(csv_path)
    assert counts == {"Training": 28, "PublicTest": 7, "PrivateTest": 7}
    dataset = FER2013Dataset(csv_path, split="Training")
    assert len(dataset) == 28
    image, label = dataset[0]
    assert image.shape == (1, 48, 48)
    assert 0 <= label < 7
    with pytest.raises(ValueError):
        generate_synth_dataset(tmp_path / "bad", samples_per_class=2)


def test_train_transform_shapes() -> None:
    image = torch.rand(1, 48, 48)
    transform = TrainTransform(image_size=48, out_chans=1, mean=[0.5], std=[0.5])
    output = transform(image)
    assert output.shape == (1, 48, 48)
    assert torch.isfinite(output).all()

    transform_rgb = TrainTransform(image_size=48, out_chans=3, mean=[0.5], std=[0.5])
    assert transform_rgb(image).shape == (3, 48, 48)


def test_eval_transform_is_deterministic() -> None:
    image = torch.full((1, 48, 48), 0.5)
    transform = EvalTransform(out_chans=1, mean=[0.5], std=[0.5])
    first = transform(image)
    second = transform(image)
    assert torch.allclose(first, second)
    assert torch.allclose(first, torch.zeros_like(first), atol=1e-6)
