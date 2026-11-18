"""emoji：基于深度学习的人脸表情识别（FER）实验骨架。

提供数据管线、模型（ResNet18 / MobileNetV3 / CBAM）、训练与评估流程、
鲁棒性与 Grad-CAM 分析，以及基于 OpenCV/MediaPipe 的实时演示原型。
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__", "main"]


def main() -> None:
    """控制台入口（pyproject 中的脚本 ``emoji`` 实际指向 ``emoji.cli:main``）。"""
    from emoji.cli import main as cli_main

    raise SystemExit(cli_main())
