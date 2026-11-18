"""项目配置：dataclass 定义、YAML 加载与深度合并、命令行覆盖。

设计要点：
- 所有超参数集中在 :class:`Config` 中，实验时保存 resolved config 即可复现；
- YAML 支持 ``_base_`` 字段递归继承（路径相对当前 YAML 文件所在目录）；
- 命令行 ``--set a.b=value`` 用 YAML 解析 value，从而保留 int/float/bool/list 类型；
- 未知字段直接报错，避免拼写错误被静默忽略。
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

# FER2013 官方 7 类表情顺序（CSV 中 emotion 标签 0..6 与之对应）
DEFAULT_CLASS_NAMES: list[str] = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]

# FER2013 CSV 中 Usage 列的三种合法取值
VALID_SPLITS: tuple[str, ...] = ("Training", "PublicTest", "PrivateTest")
VALID_LOSSES: tuple[str, ...] = ("ce", "lsce", "focal", "cbce")
VALID_CLASS_WEIGHTS: tuple[str, ...] = ("none", "balanced", "effective")
VALID_OPTIMIZERS: tuple[str, ...] = ("adamw", "sgd")
VALID_SCHEDULERS: tuple[str, ...] = ("cosine", "none")
VALID_DEVICES: tuple[str, ...] = ("auto", "cpu", "cuda", "mps")


@dataclass
class DataConfig:
    """数据与预处理相关配置。"""

    # task 目前仅保留 "fer2013"（合成数据也写成 FER2013 CSV 格式，便于复用同一 Dataset）
    task: str = "fer2013"
    root: str = "data/raw/fer2013"
    csv_path: str = "data/raw/fer2013/fer2013.csv"
    # 模型输入边长；FER2013 原图为 48×48，可选 96 做上采样消融
    image_size: int = 48
    in_chans: int = 1
    # 是否把灰度复制成 3 通道（方便复用 ImageNet 预训练权重）
    gray_to_rgb: bool = False
    num_classes: int = 7
    class_names: list[str] = field(default_factory=lambda: list(DEFAULT_CLASS_NAMES))
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = True
    # 是否对训练集施加数据增强（消融实验可通过 data.augment=false 关闭）
    augment: bool = True
    # FER2013 官方划分：PublicTest 作为验证集，PrivateTest 只在最终评估时使用
    val_split: str = "PublicTest"
    test_split: str = "PrivateTest"
    # 归一化统计量；长度为 1 时对所有通道广播，长度为 3 时逐通道
    mean: list[float] = field(default_factory=lambda: [0.5])
    std: list[float] = field(default_factory=lambda: [0.5])

    @property
    def model_in_chans(self) -> int:
        """模型实际输入通道数（gray_to_rgb 时强制 3 通道）。"""
        return 3 if self.gray_to_rgb else self.in_chans


@dataclass
class ModelConfig:
    """模型结构配置。"""

    # resnet18_fer | mobilenet_v3_small_fer | mobilenet_v3_large_fer
    name: str = "resnet18_fer"
    num_classes: int = 7
    dropout: float = 0.2
    # 默认不加载 ImageNet 预训练权重（离线可运行）；开启时需要联网下载
    pretrained: bool = False
    use_cbam: bool = False
    cbam_reduction: int = 16
    cbam_kernel_size: int = 7
    # BatchNorm 动量：torchvision MobileNetV3 默认为 0.01（长时间训练更稳），
    # 但小数据/1~2 epoch 冒烟时收敛太慢会导致 eval 模式指标异常，故统一设为 0.1
    bn_momentum: float = 0.1
    # 训练前用训练集前 N 个 batch 校准 BN running stats（0=关闭）。
    # 小数据/短训练时能避免 eval 模式统计量偏移，正式实验保持 0 即可。
    bn_warmup_batches: int = 0


@dataclass
class TrainConfig:
    """训练配置。"""

    epochs: int = 30
    optimizer: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9  # 仅 sgd 使用
    scheduler: str = "cosine"  # cosine | none
    warmup_epochs: int = 1
    min_lr_ratio: float = 0.02  # 余弦退火的最低学习率比例
    loss: str = "ce"  # ce | lsce | focal | cbce
    label_smoothing: float = 0.05
    focal_gamma: float = 2.0
    class_weight: str = "none"  # none | balanced | effective
    effective_beta: float = 0.999
    grad_clip: float = 5.0
    amp: str = "auto"  # auto | off；CPU 上自动关闭
    early_stop_patience: int = 8  # <=0 表示禁用早停
    early_stop_monitor: str = "macro_f1"  # macro_f1 | val_loss
    device: str = "auto"  # auto | cpu | cuda | mps
    deterministic: bool = True
    torch_num_threads: int = 0  # >0 时设置 CPU 线程数（0 表示保持默认）
    log_interval: int = 0  # 预留：>0 时按 step 打印，0 表示每 epoch 一行


@dataclass
class EvalConfig:
    """评估与可视化配置。"""

    split: str = "PrivateTest"
    batch_size: int = 128
    device: str = "auto"
    # Grad-CAM 目标层；留空表示自动选择最后一层卷积
    target_layer: str = ""
    top_k: int = 8  # Grad-CAM 保存的错误/正确样本数


@dataclass
class RobustnessConfig:
    """鲁棒性实验配置。"""

    split: str = "PrivateTest"
    severities: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    batch_size: int = 128
    seed: int = 42


@dataclass
class Config:
    """顶层配置：包含数据、模型、训练、评估、鲁棒性五个子配置。"""

    seed: int = 42
    output_dir: str = "outputs"
    run_name: str | None = None
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)

    # ------------------------------------------------------------------ #
    # 序列化 / 反序列化
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """转为嵌套普通 dict（用于 YAML/JSON 落盘）。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Config:
        """从嵌套 dict 构建配置，未知字段会报错。"""
        obj = _build_dataclass(cls, dict(data))
        assert isinstance(obj, Config)
        return obj

    def copy(self) -> Config:
        """深拷贝（用于在评估阶段安全地覆盖字段）。"""
        obj = Config.from_dict(self.to_dict())
        obj.validate()
        return obj

    def validate(self) -> None:
        """检查配置合法性，给出可操作的错误信息。"""
        d, m, t, e, r = self.data, self.model, self.train, self.eval, self.robustness

        if d.image_size <= 0:
            raise ValueError(f"data.image_size 必须为正整数，当前为 {d.image_size}")
        if d.in_chans not in (1, 3):
            raise ValueError(f"data.in_chans 只能是 1 或 3，当前为 {d.in_chans}")
        if d.num_classes <= 0:
            raise ValueError("data.num_classes 必须为正整数")
        if len(d.class_names) < d.num_classes:
            raise ValueError(
                f"data.class_names 长度({len(d.class_names)}) 小于 num_classes({d.num_classes})"
            )
        if m.num_classes != d.num_classes:
            raise ValueError(
                f"model.num_classes({m.num_classes}) 与 data.num_classes({d.num_classes}) 不一致"
            )
        for split_name, split in (("data.val_split", d.val_split), ("data.test_split", d.test_split)):
            if split not in VALID_SPLITS:
                raise ValueError(f"{split_name} 必须是 {VALID_SPLITS} 之一，当前为 {split!r}")
        if len(d.mean) not in (1, d.model_in_chans) or len(d.std) not in (1, d.model_in_chans):
            raise ValueError(
                "data.mean/data.std 长度必须为 1 或与模型输入通道数 "
                f"({d.model_in_chans}) 一致，当前 mean={d.mean}, std={d.std}"
            )
        if any(s <= 0 for s in d.std):
            raise ValueError(f"data.std 必须为正数，当前为 {d.std}")
        if d.batch_size <= 0:
            raise ValueError("data.batch_size 必须为正整数")

        if not 0.0 < m.bn_momentum <= 1.0:
            raise ValueError(f"model.bn_momentum 必须在 (0, 1] 内，当前为 {m.bn_momentum}")
        if m.bn_warmup_batches < 0:
            raise ValueError("model.bn_warmup_batches 不能为负")
        if t.epochs <= 0:
            raise ValueError("train.epochs 必须为正整数")
        if t.optimizer not in VALID_OPTIMIZERS:
            raise ValueError(f"train.optimizer 必须是 {VALID_OPTIMIZERS} 之一")
        if t.scheduler not in VALID_SCHEDULERS:
            raise ValueError(f"train.scheduler 必须是 {VALID_SCHEDULERS} 之一")
        if t.loss not in VALID_LOSSES:
            raise ValueError(f"train.loss 必须是 {VALID_LOSSES} 之一")
        if t.class_weight not in VALID_CLASS_WEIGHTS:
            raise ValueError(f"train.class_weight 必须是 {VALID_CLASS_WEIGHTS} 之一")
        if t.early_stop_monitor not in ("macro_f1", "val_loss"):
            raise ValueError("train.early_stop_monitor 必须是 macro_f1 或 val_loss")
        if t.device not in VALID_DEVICES:
            raise ValueError(f"train.device 必须是 {VALID_DEVICES} 之一")
        if t.warmup_epochs < 0:
            raise ValueError("train.warmup_epochs 不能为负")
        if not 0.0 <= t.label_smoothing < 1.0:
            raise ValueError("train.label_smoothing 必须在 [0, 1) 内")
        if t.focal_gamma < 0:
            raise ValueError("train.focal_gamma 不能为负")

        if e.device not in VALID_DEVICES:
            raise ValueError(f"eval.device 必须是 {VALID_DEVICES} 之一")
        if e.split not in VALID_SPLITS:
            raise ValueError(f"eval.split 必须是 {VALID_SPLITS} 之一")
        if e.batch_size <= 0:
            raise ValueError("eval.batch_size 必须为正整数")
        if e.top_k <= 0:
            raise ValueError("eval.top_k 必须为正整数")
        if not r.severities or any(s not in (1, 2, 3, 4, 5) for s in r.severities):
            raise ValueError("robustness.severities 必须是 1..5 的非空子集")


# ---------------------------------------------------------------------- #
# YAML 读写、深度合并、命令行覆盖
# ---------------------------------------------------------------------- #
def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """递归合并两个映射，``override`` 中的标量覆盖 ``base`` 中的同名值。"""
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml_file(path: Path) -> dict[str, Any]:
    """读取一个 YAML 文件，并递归展开其中的 ``_base_`` 继承。"""
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"配置文件的顶层必须是映射: {path}")
    data = dict(raw)
    bases = data.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    merged: dict[str, Any] = {}
    for base in bases:
        if not isinstance(base, str):
            raise TypeError(f"_base_ 中的路径必须是字符串: {base!r}（文件 {path}）")
        merged = deep_merge(merged, _load_yaml_file((path.parent / base).resolve()))
    return deep_merge(merged, data)


def load_config_files(paths: Sequence[str | Path]) -> dict[str, Any]:
    """按顺序加载多个 YAML 配置并深度合并（后者覆盖前者）。"""
    merged: dict[str, Any] = {}
    for path in paths:
        merged = deep_merge(merged, _load_yaml_file(Path(path)))
    return merged


def parse_override(item: str) -> tuple[list[str], Any]:
    """解析 ``--set a.b=value`` 形式的覆盖项，返回 (键路径, 值)。"""
    if "=" not in item:
        raise ValueError(f"--set 需要 key=value 形式，收到: {item!r}")
    raw_key, raw_value = item.split("=", 1)
    keys = [part for part in raw_key.strip().split(".") if part]
    if not keys:
        raise ValueError(f"--set 的 key 不能为空: {item!r}")
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError:  # 无法按 YAML 解析时退化为原始字符串
        value = raw_value
    if isinstance(value, str):
        # PyYAML 不把 "1e-3" 这类科学记数法解析为 float，这里显式兜底
        try:
            value = float(value)
        except ValueError:
            pass
    return keys, value


def apply_overrides(data: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """把 ``--set`` 覆盖项应用到嵌套 dict 上（原地修改副本）。"""
    result: dict[str, Any] = copy.deepcopy(dict(data))
    for item in overrides:
        keys, value = parse_override(item)
        cursor: dict[str, Any] = result
        for key in keys[:-1]:
            child = cursor.get(key)
            if not isinstance(child, dict):
                child = {}
                cursor[key] = child
            cursor = child
        cursor[keys[-1]] = value
    return result


def _build_dataclass(cls: type, data: Mapping[str, Any]) -> Any:
    """按 dataclass 定义递归构建实例；未知字段报错。"""
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} 不是 dataclass")
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = sorted(k for k in data if k not in known and not str(k).startswith("_"))
    if unknown:
        raise ValueError(
            f"配置中存在未知字段 {unknown}（{cls.__name__}）；"
            f"可用字段: {sorted(known)}"
        )
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        field_type = hints.get(f.name)
        if isinstance(field_type, type) and is_dataclass(field_type):
            if not isinstance(value, Mapping):
                raise ValueError(f"配置字段 {f.name} 需要映射，收到 {type(value).__name__}")
            kwargs[f.name] = _build_dataclass(field_type, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


# --smoke 时应用的快速覆盖项：合成数据 + 1 epoch + 轻量模型
SMOKE_OVERRIDES: dict[str, Any] = {
    "run_name": "smoke",
    "data": {
        "task": "fer2013",
        "csv_path": "data/synth/smoke/fer2013.csv",
        "batch_size": 8,
        "num_workers": 0,
        "pin_memory": False,
        "mean": [0.5],
        "std": [0.5],
    },
    "model": {"name": "mobilenet_v3_small_fer", "pretrained": False},
    "train": {"epochs": 1, "warmup_epochs": 0, "early_stop_patience": 2, "torch_num_threads": 2},
    "eval": {"batch_size": 16},
}


def make_config(
    config_paths: Sequence[str | Path] = (),
    overrides: Sequence[str] = (),
    smoke: bool = False,
) -> Config:
    """加载配置：YAML 文件（可多个）→ smoke 覆盖 → ``--set`` 覆盖 → 校验。"""
    data = load_config_files(config_paths)
    if smoke:
        data = deep_merge(data, SMOKE_OVERRIDES)
    data = apply_overrides(data, overrides)
    cfg = Config.from_dict(data)
    cfg.validate()
    return cfg


def save_config(cfg: Config, path: str | Path) -> Path:
    """把解析后的配置以 YAML 保存到 ``path``。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return target


def load_config(path: str | Path) -> Config:
    """从单个 YAML 文件加载配置（含 ``_base_`` 继承与校验）。"""
    return make_config([path])
