"""模型构建入口：按配置名构建 ResNet18 / MobileNetV3 的 FER 变体。"""

from __future__ import annotations

from typing import Final

from torch import nn

from emoji.config import Config, ModelConfig
from emoji.models.fer_mobilenet import FERMobileNetV3
from emoji.models.fer_resnet import FERResNet18

# 支持的模型名（含常见简写）
RESNET_NAMES: Final = ("resnet18_fer", "resnet18")
MOBILENET_SMALL_NAMES: Final = ("mobilenet_v3_small_fer", "mobilenet_v3_small")
MOBILENET_LARGE_NAMES: Final = ("mobilenet_v3_large_fer", "mobilenet_v3_large")


def _set_batch_norm_momentum(model: nn.Module, momentum: float) -> None:
    """统一设置所有 BatchNorm2d 的动量（见 ModelConfig.bn_momentum 的说明）。"""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = momentum


def build_model(model_cfg: ModelConfig, input_size: int, in_chans: int) -> nn.Module:
    """根据模型配置构建网络。

    Args:
        model_cfg: 模型结构配置。
        input_size: 输入边长（决定 ResNet18 使用改造 stem 还是标准 stem）。
        in_chans: 实际输入通道数（灰度 1 或 RGB 3）。

    Raises:
        ValueError: 模型名不被支持。
        RuntimeError: 请求了 ImageNet 预训练权重但下载失败（离线环境）。
    """
    name = model_cfg.name.lower()
    common = {
        "num_classes": model_cfg.num_classes,
        "in_chans": in_chans,
        "input_size": input_size,
        "use_cbam": model_cfg.use_cbam,
        "cbam_reduction": model_cfg.cbam_reduction,
        "cbam_kernel_size": model_cfg.cbam_kernel_size,
        "dropout": model_cfg.dropout,
    }
    try:
        if name in RESNET_NAMES:
            model: nn.Module = FERResNet18(pretrained=model_cfg.pretrained, **common)
        elif name in MOBILENET_SMALL_NAMES:
            model = FERMobileNetV3(variant="small", pretrained=model_cfg.pretrained, **common)
        elif name in MOBILENET_LARGE_NAMES:
            model = FERMobileNetV3(variant="large", pretrained=model_cfg.pretrained, **common)
        else:
            supported = RESNET_NAMES + MOBILENET_SMALL_NAMES + MOBILENET_LARGE_NAMES
            raise ValueError(f"不支持的模型 {model_cfg.name!r}；可选: {list(supported)}")
    except RuntimeError as exc:
        # torchvision 下载预训练权重失败（离线）时给出可操作的错误信息
        if model_cfg.pretrained:
            raise RuntimeError(
                f"加载 ImageNet 预训练权重失败（需要联网下载）: {exc}\n"
                "离线环境请设置 model.pretrained=false，或先手动缓存 torchvision 权重。"
            ) from exc
        raise
    _set_batch_norm_momentum(model, model_cfg.bn_momentum)
    return model


def build_model_from_config(cfg: Config) -> nn.Module:
    """直接根据完整配置构建模型。"""
    return build_model(cfg.model, cfg.data.image_size, cfg.data.model_in_chans)
