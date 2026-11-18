"""MobileNetV3（small/large）FER 变体。

改造点：
1. 首层卷积通道数适配（1/3 通道）；无预训练权重时用 Kaiming 初始化，
   有预训练权重时把 ImageNet 首层权重按通道求均值。
2. 分类头改为 ``num_classes`` 类（7 类表情）。
3. 可选 CBAM：采用标准位置——对 MobileNetV3 的残差块（``InvertedResidual``），
   把 CBAM 接在**卷积分支输出之后、残差相加之前**；对无残差的 block（如 stem
   卷积），CBAM 直接接在卷积输出之后。CBAM 不改变块输出尺寸，便于与基线公平对比。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from emoji.models.cbam import CBAM

_VARIANT_ALIASES = {
    "small": "small",
    "mobilenet_v3_small": "small",
    "mobilenet_v3_small_fer": "small",
    "large": "large",
    "mobilenet_v3_large": "large",
    "mobilenet_v3_large_fer": "large",
}


def _create_backbone(variant: str) -> nn.Module:
    """创建 torchvision MobileNetV3 骨干（weights=None，不下载权重）。"""
    try:
        from torchvision.models import mobilenet_v3_large, mobilenet_v3_small
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("未安装 torchvision，无法构建 MobileNetV3") from exc
    if variant == "small":
        return mobilenet_v3_small(weights=None)
    if variant == "large":
        return mobilenet_v3_large(weights=None)
    raise ValueError(f"未知 MobileNetV3 变体 {variant!r}")


def _infer_block_outputs(net: nn.Module, in_chans: int, input_size: int) -> list[tuple[int, int, int]]:
    """dry-run 一次，记录 ``net.features`` 每个 block 的输出 (C, H, W)。"""
    stats: list[tuple[int, int, int]] = []

    def _hook(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
        stats.append((int(output.shape[1]), int(output.shape[2]), int(output.shape[3])))

    handles = [block.register_forward_hook(_hook) for block in net.features]
    was_training = net.training
    net.eval()
    with torch.no_grad():
        net(torch.zeros(1, in_chans, input_size, input_size))
    net.train(was_training)
    for handle in handles:
        handle.remove()
    return stats


def _apply_cbam(
    net: nn.Module,
    in_chans: int,
    input_size: int,
    reduction: int,
    kernel_size: int,
) -> None:
    """在残差块内部插入 CBAM：卷积输出之后、残差相加之前。

    torchvision 的 ``InvertedResidual`` 把卷积分支封装在 ``block`` 子模块中，
    其 ``forward`` 形式为 ``x + self.block(x)``（有残差）或 ``self.block(x)``
    （无残差）。因此把 CBAM 追加到 ``block`` 末尾，即可保证"卷积之后、残差相加
    之前"的标准 CBAM 位置；而 stem 这类无残差 block 则把 CBAM 接在输出之后。
    """
    stats = _infer_block_outputs(net, in_chans, input_size)
    new_features = nn.Sequential()
    for index, block in enumerate(net.features):
        channels = stats[index][0]
        cbam = CBAM(channels, reduction=reduction, kernel_size=kernel_size)
        inner = getattr(block, "block", None)
        if isinstance(inner, nn.Sequential) and hasattr(block, "use_res_connect"):
            # 残差块：CBAM 位于卷积分支末端，残差相加之前
            block.block = nn.Sequential(inner, cbam)
            new_features.add_module(f"block{index}", block)
        else:
            # 无残差块（stem 等）：CBAM 直接接在卷积输出之后
            new_features.add_module(f"block{index}", nn.Sequential(block, cbam))
    net.features = new_features


class FERMobileNetV3(nn.Module):
    """FER 版 MobileNetV3（small 或 large），支持 1/3 通道、7 类与可选 CBAM。"""

    def __init__(
        self,
        variant: str = "small",
        num_classes: int = 7,
        in_chans: int = 1,
        input_size: int = 48,
        use_cbam: bool = False,
        cbam_reduction: int = 16,
        cbam_kernel_size: int = 7,
        dropout: float | None = 0.2,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        variant_key = _VARIANT_ALIASES.get(str(variant).lower())
        if variant_key is None:
            raise ValueError(
                f"未知 MobileNetV3 变体 {variant!r}；可用: {sorted(set(_VARIANT_ALIASES))}"
            )
        if in_chans not in (1, 3):
            raise ValueError(f"in_chans 只能是 1 或 3，当前为 {in_chans}")
        self.variant = variant_key
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.input_size = input_size
        self.use_cbam = use_cbam

        net = _create_backbone(variant_key)
        self._adapt_first_conv(net, in_chans)
        self._replace_classifier_head(net, num_classes, dropout)

        # 预训练权重必须在插入 CBAM 之前加载（否则 features 的模块名会变化）
        if pretrained:
            self.net = net
            self.load_imagenet_weights()
        else:
            self.net = net

        if use_cbam:
            _apply_cbam(
                self.net,
                in_chans,
                input_size,
                cbam_reduction,
                cbam_kernel_size,
            )

    # ------------------------------------------------------------------ #
    # 结构改造
    # ------------------------------------------------------------------ #
    @staticmethod
    def _adapt_first_conv(net: nn.Module, in_chans: int) -> None:
        """把首层 3 通道卷积替换为 ``in_chans`` 通道，并迁移已有权重。"""
        old_conv = net.features[0][0]
        if not isinstance(old_conv, nn.Conv2d):
            raise TypeError("MobileNetV3 首层结构异常：features[0][0] 不是 Conv2d")
        if old_conv.in_channels == in_chans:
            return
        new_conv = nn.Conv2d(
            in_chans,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        if new_conv.bias is not None and old_conv.bias is not None:
            new_conv.bias.data.copy_(old_conv.bias.data)
        if in_chans == 1:
            # 灰度为 RGB 权重均值（与 FERResNet18 的适配方式一致）
            new_conv.weight.data.copy_(old_conv.weight.data.mean(dim=1, keepdim=True))
        elif in_chans != old_conv.in_channels:
            raise ValueError(f"不支持的输入通道数 {in_chans}")
        net.features[0][0] = new_conv

    @staticmethod
    def _replace_classifier_head(net: nn.Module, num_classes: int, dropout: float | None) -> None:
        """替换分类头最后一层为 ``num_classes`` 维，并按需覆盖 Dropout。"""
        last = net.classifier[-1]
        if not isinstance(last, nn.Linear):
            raise TypeError("MobileNetV3 分类头结构异常：classifier[-1] 不是 Linear")
        net.classifier[-1] = nn.Linear(last.in_features, num_classes)
        if dropout is not None:
            for index, module in enumerate(net.classifier):
                if isinstance(module, nn.Dropout):
                    net.classifier[index] = nn.Dropout(dropout)

    # ------------------------------------------------------------------ #
    # 前向与可视化辅助
    # ------------------------------------------------------------------ #
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def gradcam_target_layer(self) -> nn.Module:
        """选择空间分辨率仍不小于 3×3 的最后一个卷积层（CAM 图更可读）。

        若不存在（输入太小），退化为网络中的最后一个卷积层。
        """
        convs = [module for module in self.net.modules() if isinstance(module, nn.Conv2d)]
        target: nn.Module | None = None

        def _hook(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
            nonlocal target
            if output.shape[2] >= 3 and output.shape[3] >= 3:
                target = _module

        handles = [conv.register_forward_hook(_hook) for conv in convs]
        was_training = self.training
        self.eval()
        with torch.no_grad():
            self.net(torch.zeros(1, self.in_chans, self.input_size, self.input_size))
        self.train(was_training)
        for handle in handles:
            handle.remove()
        if target is None:
            target = convs[-1]
        return target

    def feature_channels(self) -> int:
        """分类头输入维度。"""
        return int(self.net.classifier[-1].in_features)

    def load_imagenet_weights(self) -> None:
        """加载 torchvision ImageNet 预训练权重（需要联网下载）。"""
        try:
            from torchvision.models import (
                MobileNet_V3_Large_Weights,
                MobileNet_V3_Small_Weights,
                mobilenet_v3_large,
                mobilenet_v3_small,
            )
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("未安装 torchvision，无法加载 ImageNet 预训练权重") from exc

        if self.variant == "small":
            reference = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        else:
            reference = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        ref_state = {f"net.{key}": value for key, value in reference.state_dict().items()}
        first_key = "net.features.0.0.weight"
        if first_key in ref_state and self.in_chans == 1:
            ref_state[first_key] = ref_state[first_key].mean(dim=1, keepdim=True)

        own_state = self.state_dict()
        compatible = {
            key: value
            for key, value in ref_state.items()
            if key in own_state and own_state[key].shape == value.shape
        }
        missing = [key for key in own_state if key not in compatible]
        self.load_state_dict(compatible, strict=False)
        if missing:
            print(f"[models] ImageNet 预训练：{len(missing)} 个权重因形状/名称不匹配保持随机初始化")
