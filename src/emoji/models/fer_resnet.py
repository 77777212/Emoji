"""面向 48×48 灰度表情图的 ResNet18 变体（可开关 CBAM）。

与标准 ImageNet ResNet18 的两点关键差异：

1. **Stem 适配**（输入 48×48 时）：标准 stem 的 7×7/stride2 卷积 + stride2 最大池化
   会把 48×48 先降到 12×12，再经 4 个 stage（总 stride 32）后只剩 1×1，特征图过小、
   精度明显受损。因此当 ``input_size <= 64`` 时使用 3×3/stride1 卷积并去掉 maxpool，
   末端特征图为 6×6；输入 96/224 时使用标准 stem（末端 3×3/7×7）。
2. **CBAM 插入位置**：采用标准做法——在**每个 BasicBlock 内部**，第二个卷积
   （``conv2`` + ``bn2``）之后、与 identity 残差相加之前应用 CBAM：

   ``out = bn2(conv2(relu(bn1(conv1(x)))))`` → ``out = CBAM(out)`` → ``relu(out + identity)``

   CBAM 不改变通道数与空间尺寸，因此与基线结构完全对齐；这比"整个 stage 输出端再做
   一次注意力"更符合 CBAM 原论文的用法（在残差分支内部抑制/增强特征）。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from emoji.models.cbam import CBAM


class BasicBlock(nn.Module):
    """ResNet18 的残差基本块（两个 3×3 卷积 + 恒等/投影捷径）。

    ``use_cbam=True`` 时，在 ``conv2``/``bn2`` 之后、残差相加之前插入 CBAM。
    """

    expansion = 1

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        use_cbam: bool = False,
        cbam_reduction: int = 16,
        cbam_kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_chans, out_chans, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chans)
        self.downsample = downsample
        # 标准 CBAM 位置：残差分支内部、卷积输出之后
        self.cbam = (
            CBAM(out_chans, reduction=cbam_reduction, kernel_size=cbam_kernel_size)
            if use_cbam
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.cbam is not None:
            out = self.cbam(out)  # 卷积之后、残差相加之前
        return self.relu(out + identity)


def _make_stage(
    in_chans: int,
    out_chans: int,
    stride: int,
    cbam_reduction: int,
    cbam_kernel_size: int,
    use_cbam: bool,
) -> nn.Sequential:
    """构建一个 stage（2 个 BasicBlock），每个 block 内部按需插入 CBAM。"""
    downsample = None
    if stride != 1 or in_chans != out_chans:
        downsample = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_chans),
        )
    return nn.Sequential(
        BasicBlock(
            in_chans,
            out_chans,
            stride,
            downsample,
            use_cbam=use_cbam,
            cbam_reduction=cbam_reduction,
            cbam_kernel_size=cbam_kernel_size,
        ),
        BasicBlock(
            out_chans,
            out_chans,
            1,
            None,
            use_cbam=use_cbam,
            cbam_reduction=cbam_reduction,
            cbam_kernel_size=cbam_kernel_size,
        ),
    )


class FERResNet18(nn.Module):
    """FER 版 ResNet18：支持 1/3 通道、48（改造 stem）/96（标准 stem）、可选 CBAM。"""

    def __init__(
        self,
        num_classes: int = 7,
        in_chans: int = 1,
        input_size: int = 48,
        use_cbam: bool = False,
        cbam_reduction: int = 16,
        cbam_kernel_size: int = 7,
        dropout: float = 0.2,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        if in_chans not in (1, 3):
            raise ValueError(f"in_chans 只能是 1 或 3，当前为 {in_chans}")
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.input_size = input_size
        self.use_cbam = use_cbam

        if input_size <= 64:
            # 小图（48×48）：3×3/stride1 + 无 maxpool，保证末端特征至少 4×4
            self.conv1 = nn.Conv2d(in_chans, 64, 3, stride=1, padding=1, bias=False)
            self.maxpool = nn.Identity()
        else:
            # 大图（96/224）：标准 ImageNet stem
            self.conv1 = nn.Conv2d(in_chans, 64, 7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        stage_specs = [(64, 1), (128, 2), (256, 2), (512, 2)]
        in_stage = 64
        self.layer1, in_stage = self._build_stage(
            in_stage, *stage_specs[0], cbam_reduction, cbam_kernel_size, use_cbam
        )
        self.layer2, in_stage = self._build_stage(
            in_stage, *stage_specs[1], cbam_reduction, cbam_kernel_size, use_cbam
        )
        self.layer3, in_stage = self._build_stage(
            in_stage, *stage_specs[2], cbam_reduction, cbam_kernel_size, use_cbam
        )
        self.layer4, _ = self._build_stage(
            in_stage, *stage_specs[3], cbam_reduction, cbam_kernel_size, use_cbam
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512, num_classes)

        self._init_weights()
        if pretrained:
            self.load_imagenet_weights()

    @staticmethod
    def _build_stage(
        in_chans: int,
        out_chans: int,
        stride: int,
        cbam_reduction: int,
        cbam_kernel_size: int,
        use_cbam: bool,
    ) -> tuple[nn.Sequential, int]:
        stage = _make_stage(
            in_chans,
            out_chans,
            stride,
            cbam_reduction,
            cbam_kernel_size,
            use_cbam,
        )
        return stage, out_chans

    def _init_weights(self) -> None:
        """Kaiming 初始化卷积、BN 置为恒等；从零训练时使用。"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

    # ------------------------------------------------------------------ #
    # 推理/可视化辅助
    # ------------------------------------------------------------------ #
    def gradcam_target_layer(self) -> nn.Module:
        """Grad-CAM 目标层：最后一个卷积 stage 的输出。"""
        return self.layer4

    def feature_channels(self) -> int:
        """全局池化后的特征维度。"""
        return 512

    def load_imagenet_weights(self) -> None:
        """加载 torchvision ImageNet 预训练权重（需要联网下载）。

        自动适配：首层通道数（灰度取 RGB 权重均值）、以及形状不匹配的层
        （例如 48×48 输入的 stem 卷积、分类头）。形状匹配的层会严格拷贝。
        """
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as exc:  # pragma: no cover - 正常环境一定可用
            raise RuntimeError("未安装 torchvision，无法加载 ImageNet 预训练权重") from exc

        reference = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        ref_state = reference.state_dict()
        conv1_weight = ref_state["conv1.weight"]
        if self.in_chans == 1:
            ref_state["conv1.weight"] = conv1_weight.mean(dim=1, keepdim=True)
        elif self.in_chans != 3:
            raise ValueError(f"不支持的 in_chans={self.in_chans}")

        own_state = self.state_dict()
        compatible = {
            key: value
            for key, value in ref_state.items()
            if key in own_state and own_state[key].shape == value.shape
        }
        missing = [key for key in own_state if key not in compatible]
        self.load_state_dict(compatible, strict=False)
        if missing:
            print(f"[models] ImageNet 预训练：{len(missing)} 个权重因形状不匹配而随机初始化")
