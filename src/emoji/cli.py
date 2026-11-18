"""命令行入口：``emoji`` 脚本。

子命令：
- ``synth-data``：生成合成 FER 数据集（无真实数据时冒烟/测试用）；
- ``train``：训练模型（支持 YAML 配置继承、``--set`` 覆盖、``--smoke``）；
- ``eval``：加载 checkpoint 在 public/private 等划分上评估；
- ``robustness``：噪声/擦除/降分辨率鲁棒性实验；
- ``gradcam``：错误/正确样本的 Grad-CAM 可视化；
- ``demo``：摄像头实时表情识别演示（需要 ``uv sync --extra demo``）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from emoji import __version__
from emoji.config import make_config


def build_parser() -> argparse.ArgumentParser:
    """构建 argparse 解析器（子命令 + 全局帮助）。"""
    parser = argparse.ArgumentParser(
        prog="emoji",
        description="基于深度学习的人脸表情识别：训练、评估、鲁棒性、Grad-CAM 与实时演示",
    )
    parser.add_argument("--version", action="version", version=f"emoji {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---------------------------------------------------------------- #
    # synth-data
    # ---------------------------------------------------------------- #
    synth = subparsers.add_parser(
        "synth-data", help="生成合成 FER 数据集（与 FER2013 CSV 同格式）"
    )
    synth.add_argument("--out", default="data/synth/smoke", help="输出目录")
    synth.add_argument("--samples-per-class", type=int, default=32, help="每类样本数（>=3）")
    synth.add_argument("--image-size", type=int, default=48, help="图像边长（默认 48）")
    synth.add_argument("--seed", type=int, default=0, help="随机种子")
    synth.add_argument("--train-frac", type=float, default=0.7, help="训练集占比")
    synth.add_argument("--val-frac", type=float, default=0.15, help="验证集占比")

    # ---------------------------------------------------------------- #
    # 训练 / 评估 / 鲁棒性 / Grad-CAM 的公共参数
    # ---------------------------------------------------------------- #
    def add_set_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--set",
            dest="overrides",
            action="append",
            default=[],
            help="覆盖配置项，例如 --set train.lr=1e-3 --set model.use_cbam=true",
        )

    def add_config_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--config",
            action="append",
            default=[],
            help="YAML 配置文件（可多次指定，后者覆盖前者）",
        )
        add_set_args(sub)

    train = subparsers.add_parser("train", help="训练表情识别模型")
    add_config_args(train)
    train.add_argument("--smoke", action="store_true", help="冒烟模式：合成数据 + 1 epoch + 轻量模型")
    train.add_argument("--resume", default=None, help="从 last.pt 恢复训练")
    train.add_argument("--run-name", default=None, help="覆盖输出目录名")

    eval_parser = subparsers.add_parser("eval", help="在指定数据划分上评估 checkpoint")
    add_set_args(eval_parser)
    eval_parser.add_argument("--checkpoint", required=True, help="best.pt / last.pt 路径")
    eval_parser.add_argument("--split", default=None, help="Training/PublicTest/PrivateTest")
    eval_parser.add_argument("--out", default=None, help="输出目录（默认 <ckpt目录>/eval_<split>）")
    eval_parser.add_argument("--no-plot", action="store_true", help="不绘制混淆矩阵")
    eval_parser.add_argument("--no-timing", action="store_true", help="不统计推理时延")

    robustness = subparsers.add_parser("robustness", help="噪声/擦除/降分辨率鲁棒性实验")
    add_set_args(robustness)
    robustness.add_argument("--checkpoint", required=True, help="best.pt / last.pt 路径")
    robustness.add_argument("--split", default=None, help="测试划分（默认 PrivateTest）")
    robustness.add_argument("--out", default=None, help="输出目录")
    robustness.add_argument(
        "--severities", default=None, help="逗号分隔的强度，如 1,3,5（默认 1..5）"
    )
    robustness.add_argument(
        "--max-samples", type=int, default=0, help="最多评估样本数（0=全部，调试用）"
    )

    gradcam = subparsers.add_parser("gradcam", help="Grad-CAM 可视化")
    add_set_args(gradcam)
    gradcam.add_argument("--checkpoint", required=True, help="best.pt / last.pt 路径")
    gradcam.add_argument("--split", default=None, help="数据划分（默认 PrivateTest）")
    gradcam.add_argument("--out", default=None, help="输出目录")
    gradcam.add_argument("--top-k", type=int, default=8, help="错误/正确样本各保存张数")
    gradcam.add_argument(
        "--target-layer", default=None, help="目标层点路径，如 layer4；默认自动选择"
    )

    demo = subparsers.add_parser("demo", help="摄像头实时表情识别演示")
    demo.add_argument("--checkpoint", required=True, help="best.pt / last.pt 路径")
    demo.add_argument("--camera", type=int, default=0, help="摄像头编号")
    demo.add_argument("--width", type=int, default=640)
    demo.add_argument("--height", type=int, default=480)
    demo.add_argument("--window", type=int, default=30, help="近期表情统计的滑动窗口帧数")
    demo.add_argument("--no-mediapipe", action="store_true", help="强制使用 OpenCV Haar 检测器")
    demo.add_argument("--device", default=None, help="推理设备，默认沿用 checkpoint 配置")

    return parser


# ---------------------------------------------------------------------- #
# 各子命令实现
# ---------------------------------------------------------------------- #
def _cmd_synth_data(args: argparse.Namespace) -> int:
    from emoji.data.fer2013 import count_rows_by_usage
    from emoji.data.synth import generate_synth_dataset

    if args.samples_per_class < 3:
        raise ValueError("--samples-per-class 至少为 3")
    csv_path = generate_synth_dataset(
        args.out,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
        image_size=args.image_size,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )
    counts = count_rows_by_usage(csv_path)
    print(f"[synth-data] 已生成 {csv_path}")
    for split_name, count in counts.items():
        print(f"  {split_name}: {count} 条")
    print(f"  每类 {args.samples_per_class} 张，图像 {args.image_size}×{args.image_size}，seed={args.seed}")
    print(f"[meta] {Path(args.out) / 'meta.json'}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from emoji.data.synth import generate_synth_dataset
    from emoji.train import run_training

    cfg = make_config(args.config, args.overrides, smoke=args.smoke)
    if args.run_name:
        cfg.run_name = args.run_name
    if args.smoke and not Path(cfg.data.csv_path).is_file():
        print(f"[smoke] 未找到 {cfg.data.csv_path}，自动生成合成数据（每类 24 张）...")
        generate_synth_dataset(
            Path(cfg.data.csv_path).parent,
            samples_per_class=24,
            seed=0,
            image_size=cfg.data.image_size,
        )
    run_training(cfg, resume=args.resume)
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from emoji.evaluate import evaluate_checkpoint

    artifacts = evaluate_checkpoint(
        args.checkpoint,
        split=args.split,
        out_dir=args.out,
        overrides=args.overrides,
        save_plot=not args.no_plot,
        measure_time=not args.no_timing,
    )
    print(
        f"[eval] acc={artifacts.metrics['acc']:.4f} "
        f"macro_f1={artifacts.metrics['macro_f1']:.4f}"
    )
    print(f"[files] {artifacts.metrics_path} {artifacts.per_class_path} {artifacts.confusion_matrix_path}")
    return 0


def _cmd_robustness(args: argparse.Namespace) -> int:
    from emoji.robustness import run_robustness

    severities: list[int] | None = None
    if args.severities:
        severities = [int(part.strip()) for part in args.severities.split(",") if part.strip()]
        if not severities:
            raise ValueError("--severities 解析为空")
    run_robustness(
        args.checkpoint,
        split=args.split,
        out_dir=args.out,
        severities=severities,
        overrides=args.overrides,
        max_samples=args.max_samples,
    )
    return 0


def _cmd_gradcam(args: argparse.Namespace) -> int:
    from emoji.gradcam import save_gradcam_samples

    save_gradcam_samples(
        args.checkpoint,
        split=args.split,
        out_dir=args.out,
        top_k=args.top_k,
        overrides=args.overrides,
        target_layer=args.target_layer,
    )
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from emoji.system.demo import run_demo

    run_demo(
        args.checkpoint,
        camera=args.camera,
        width=args.width,
        height=args.height,
        window=args.window,
        prefer_mediapipe=not args.no_mediapipe,
        device=args.device,
    )
    return 0


_DISPATCH = {
    "synth-data": _cmd_synth_data,
    "train": _cmd_train,
    "eval": _cmd_eval,
    "robustness": _cmd_robustness,
    "gradcam": _cmd_gradcam,
    "demo": _cmd_demo,
}


def main(argv: list[str] | None = None) -> int:
    """命令行主函数；返回进程退出码。"""
    args = build_parser().parse_args(argv)
    handler = _DISPATCH[args.command]
    try:
        return handler(args)
    except KeyboardInterrupt:
        print("\n[中断] 已取消", file=sys.stderr)
        return 130
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
