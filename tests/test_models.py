"""模型结构测试：前向形状、stem 适配、CBAM 开关、参数量与构建入口。"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from emoji.config import ModelConfig
from emoji.models import CBAM, FERMobileNetV3, FERResNet18, build_model
from emoji.utils import count_params


def _stage4_shape(model: FERResNet18, input_size: int, in_chans: int = 1) -> tuple[int, ...]:
    """用 hook 获取 ResNet18 layer4 输出形状，验证 stem 适配是否达到预期分辨率。"""
    captured: dict[str, tuple[int, ...]] = {}
    handle = model.layer4.register_forward_hook(
        lambda _module, _inputs, output: captured.update(shape=tuple(output.shape))
    )
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, in_chans, input_size, input_size))
    handle.remove()
    return captured["shape"]


def test_resnet18_small_stem_for_48_input() -> None:
    model = FERResNet18(num_classes=7, in_chans=1, input_size=48)
    assert _stage4_shape(model, 48) == (1, 512, 6, 6)
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 1, 48, 48))
    assert logits.shape == (2, 7)
    assert torch.isfinite(logits).all()


def test_resnet18_standard_stem_for_96_input() -> None:
    model = FERResNet18(num_classes=7, in_chans=3, input_size=96)
    assert _stage4_shape(model, 96, in_chans=3) == (1, 512, 3, 3)
    model.eval()
    with torch.no_grad():
        assert model(torch.randn(1, 3, 96, 96)).shape == (1, 7)


def test_resnet18_cbam_preserves_shape_and_adds_params() -> None:
    plain = FERResNet18(num_classes=7, in_chans=1, input_size=48, use_cbam=False)
    with_cbam = FERResNet18(num_classes=7, in_chans=1, input_size=48, use_cbam=True)
    assert count_params(with_cbam) > count_params(plain)
    with_cbam.eval()
    with torch.no_grad():
        assert with_cbam(torch.randn(1, 1, 48, 48)).shape == (1, 7)


def test_cbam_module_shape_and_range() -> None:
    module = CBAM(32, reduction=16, kernel_size=7)
    inputs = torch.randn(2, 32, 12, 12)
    outputs = module(inputs)
    assert outputs.shape == inputs.shape
    assert torch.isfinite(outputs).all()


def test_resnet18_param_count_sane() -> None:
    model = FERResNet18(num_classes=7, in_chans=1, input_size=48)
    params = count_params(model)
    # 标准 ResNet18 约 11.2M；7 类分类头略小
    assert 10_000_000 < params < 15_000_000


@pytest.mark.parametrize(
    "name",
    ["resnet18_fer", "mobilenet_v3_small_fer", "mobilenet_v3_large_fer"],
)
def test_build_model_forward(name: str) -> None:
    model = build_model(ModelConfig(name=name, num_classes=7), input_size=48, in_chans=1)
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 1, 48, 48))
    assert logits.shape == (2, 7)
    assert torch.isfinite(logits).all()


def test_build_model_with_cbam_mobilenet() -> None:
    model = build_model(
        ModelConfig(name="mobilenet_v3_small_fer", num_classes=7, use_cbam=True),
        input_size=96,
        in_chans=3,
    )
    model.eval()
    with torch.no_grad():
        assert model(torch.randn(1, 3, 96, 96)).shape == (1, 7)


def test_build_model_unknown_raises() -> None:
    with pytest.raises(ValueError, match="不支持"):
        build_model(ModelConfig(name="not-a-model"), input_size=48, in_chans=1)


def test_gradcam_target_layer_available() -> None:
    for model in (
        FERResNet18(num_classes=7, in_chans=1, input_size=48),
        FERMobileNetV3(variant="small", num_classes=7, in_chans=1, input_size=48),
    ):
        layer = model.gradcam_target_layer()
        assert isinstance(layer, nn.Module)


def test_pretrained_default_off() -> None:
    """测试与冒烟过程不得下载权重，默认必须关闭预训练。"""
    assert ModelConfig().pretrained is False


def test_resnet18_cbam_is_inside_basic_blocks() -> None:
    """CBAM 应位于每个 BasicBlock 内部（conv2/bn2 之后、残差相加之前）。"""
    model = FERResNet18(num_classes=7, in_chans=1, input_size=48, use_cbam=True)
    for stage in (model.layer1, model.layer2, model.layer3, model.layer4):
        for block in stage:
            assert isinstance(block.cbam, CBAM)
    plain = FERResNet18(num_classes=7, in_chans=1, input_size=48, use_cbam=False)
    assert all(block.cbam is None for stage in (plain.layer1, plain.layer2) for block in stage)


def test_mobilenet_cbam_is_before_residual_add() -> None:
    """MobileNetV3 的 CBAM 应接在 InvertedResidual 卷积分支末端（残差相加之前）。"""
    model = FERMobileNetV3(
        variant="small", num_classes=7, in_chans=1, input_size=48, use_cbam=True
    )
    # stem（无残差块）：CBAM 直接接在 block 输出之后
    assert isinstance(model.net.features[0][-1], CBAM)
    residual_checked = 0
    for block in model.net.features:
        inner = getattr(block, "block", None)
        if isinstance(inner, nn.Sequential) and hasattr(block, "use_res_connect"):
            assert isinstance(inner[-1], CBAM)
            residual_checked += 1
    assert residual_checked >= 1


def test_batch_norm_calibration_updates_running_stats() -> None:
    """BN warmup 应更新 running stats，并在结束后恢复 momentum。"""
    from emoji.train import calibrate_batch_norm

    model = FERMobileNetV3(variant="small", num_classes=7, in_chans=1, input_size=48)
    batch_norm = model.net.features[0][1]
    assert isinstance(batch_norm, nn.BatchNorm2d)
    before = batch_norm.running_mean.clone()
    fake_loader = [
        (torch.randn(4, 1, 48, 48), torch.zeros(4, dtype=torch.long)) for _ in range(3)
    ]
    calibrate_batch_norm(model, fake_loader, torch.device("cpu"), max_batches=3)
    assert not torch.equal(batch_norm.running_mean, before)
    assert batch_norm.momentum is not None
