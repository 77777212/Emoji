"""配置系统测试：YAML 深度合并、_base_ 继承、--set 覆盖与校验。"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from emoji.config import (
    Config,
    apply_overrides,
    deep_merge,
    load_config,
    make_config,
    parse_override,
    save_config,
)


def test_deep_merge_recursive() -> None:
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    override = {"a": {"b": 10}, "e": 4}
    merged = deep_merge(base, override)
    assert merged == {"a": {"b": 10, "c": 2}, "d": 3, "e": 4}
    # 不修改输入
    assert base == {"a": {"b": 1, "c": 2}, "d": 3}


def test_parse_override_types() -> None:
    assert parse_override("train.lr=1e-3") == (["train", "lr"], 0.001)
    assert parse_override("model.use_cbam=true") == (["model", "use_cbam"], True)
    assert parse_override("seed=0") == (["seed"], 0)
    assert parse_override("data.class_names=[a, b]") == (["data", "class_names"], ["a", "b"])
    with pytest.raises(ValueError):
        parse_override("train.lr")


def test_apply_overrides_nested() -> None:
    data = {"train": {"lr": 0.1, "epochs": 10}}
    updated = apply_overrides(data, ["train.lr=0.01", "train.epochs=2"])
    assert updated["train"]["lr"] == 0.01
    assert updated["train"]["epochs"] == 2
    # 不修改输入
    assert data["train"]["lr"] == 0.1


def test_config_base_inheritance(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump({"train": {"epochs": 10, "lr": 0.1}, "model": {"name": "resnet18_fer"}}),
        encoding="utf-8",
    )
    child = tmp_path / "child.yaml"
    child.write_text(
        yaml.safe_dump({"_base_": ["base.yaml"], "train": {"epochs": 1}}),
        encoding="utf-8",
    )
    cfg = load_config(child)
    assert cfg.train.epochs == 1
    assert cfg.train.lr == pytest.approx(0.1)
    assert cfg.model.name == "resnet18_fer"


def test_config_unknown_field_raises() -> None:
    with pytest.raises(ValueError, match="未知字段"):
        make_config(overrides=["train.not_a_field=1"])


def test_config_validation_rejects_class_mismatch() -> None:
    cfg = Config.from_dict({"data": {"num_classes": 7}, "model": {"num_classes": 3}})
    with pytest.raises(ValueError, match="num_classes"):
        cfg.validate()


def test_config_validation_rejects_bad_loss() -> None:
    cfg = Config.from_dict({"train": {"loss": "mse"}})
    with pytest.raises(ValueError, match="train.loss"):
        cfg.validate()


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    cfg = make_config(overrides=["model.use_cbam=true", "train.epochs=2"])
    path = save_config(cfg, tmp_path / "resolved.yaml")
    reloaded = load_config(path)
    assert reloaded.model.use_cbam is True
    assert reloaded.train.epochs == 2
    assert reloaded.to_dict() == cfg.to_dict()


def test_smoke_overrides_are_fast() -> None:
    cfg = make_config(smoke=True)
    assert cfg.train.epochs == 1
    assert cfg.data.batch_size == 8
    assert cfg.model.name == "mobilenet_v3_small_fer"
    assert cfg.data.csv_path.endswith("data/synth/smoke/fer2013.csv")
