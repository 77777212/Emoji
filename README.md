# 基于深度学习的人脸表情识别（FER）

本仓库是本科毕业设计《基于深度学习的人脸表情识别算法研究》的**可复现实验骨架**：
以 FER2013 为主基准，复现 ResNet18 / MobileNetV3 基线，支持 CBAM 注意力改进、
Focal Loss / 类别加权等损失对比、跨数据集评估、噪声/遮挡/低分辨率鲁棒性实验、
Grad-CAM 可解释性分析与摄像头实时演示原型。

> 研究方案与进度见 [`doc/开题报告_基于深度学习的人脸表情识别算法研究.md`](<doc/开题报告_基于深度学习的人脸表情识别算法研究.md>)。
> 当前为**最小可运行骨架**：合成数据全流程已跑通；真实数据结果、跨数据集适配与
> 多 seed 汇总脚本待补（见文末 TODO）。

## 目录结构

```text
Emoji/
├── configs/                  # YAML 实验配置（支持 _base_ 继承与 --set 覆盖）
│   ├── base.yaml             # 基础配置（真实 FER2013）
│   ├── smoke.yaml            # 冒烟：合成数据 + MobileNetV3-small + 2 epoch
│   ├── baseline_resnet18.yaml
│   ├── cbam_resnet18.yaml
│   └── mobilenetv3_small.yaml
├── data/                     # 数据目录（真实数据与合成数据都不入库）
│   └── README.md             # FER2013 / RAF-DB / CK+ 获取方式与约定
├── doc/                      # 开题报告（md/docx/pdf）与技术路线图
├── src/emoji/
│   ├── cli.py                # emoji synth-data/train/eval/robustness/gradcam/demo
│   ├── config.py             # dataclass 配置 + YAML 深度合并 + --set 覆盖
│   ├── train.py              # 训练循环、余弦退火+warmup、早停、checkpoint、记录
│   ├── evaluate.py           # checkpoint 评估、每类指标、混淆矩阵、时延统计
│   ├── robustness.py         # 高斯噪声 / 随机擦除 / 降分辨率（severity 1–5）
│   ├── gradcam.py            # Grad-CAM 错误/正确样本可视化
│   ├── losses.py             # FocalLoss / LabelSmoothingCE / ClassWeightedCE
│   ├── metrics.py            # 指标计算、混淆矩阵绘图、推理时延测量
│   ├── utils.py              # seed / device / 参数量 / JSON、CSV 落盘
│   ├── data/                 # FER2013 CSV Dataset、数据增强、合成数据生成
│   ├── models/               # CBAM、FER 版 ResNet18 / MobileNetV3、模型构建
│   └── system/               # 实时演示（OpenCV/MediaPipe 延迟导入）
├── tests/                    # pytest（CPU、合成数据、< 90s）
├── pyproject.toml            # 依赖与工具配置（勿手改）
└── README.md
```

## 安装

前置：Python 3.12–3.15（仓库 `.python-version` 为 3.14）+ [uv](https://docs.astral.sh/uv/)。

```bash
# 1) 安装核心依赖（CPU 版 torch 源已在 pyproject.toml 中配置为 pytorch-cpu）
uv sync

# 2) 可选：安装实时演示依赖（opencv-python + mediapipe）
uv sync --extra demo
```

说明：

- `uv sync` 会创建/更新 `.venv` 并安装本项目为可编辑包，命令用 `uv run emoji ...`
  或直接 `.venv/bin/emoji ...`。
- **沙箱/容器中 HOME 缓存不可写**时，把 uv 缓存指到仓库内（本仓库已 gitignore `.cache/`）：

  ```bash
  UV_CACHE_DIR="$PWD/.cache/uv" uv sync
  UV_CACHE_DIR="$PWD/.cache/uv" uv run emoji --help
  ```

- 默认**不下载任何预训练权重**（`model.pretrained: false`），离线即可训练。
  开启 ImageNet 迁移学习需要网络：`--set model.pretrained=true`。
- 没有 GPU 也能跑：本项目所有实验都可用 CPU 完成（FER2013 图像仅 48×48）。

## 数据准备

真实数据需要自行下载（本仓库不包含数据，见 [`data/README.md`](data/README.md)）：

- **FER2013（必需）**：Kaggle 下载 `fer2013.csv` → 放到 `data/raw/fer2013/fer2013.csv`；
- **RAF-DB / CK+（跨数据集实验，可选）**：按 data/README.md 申请后转换为同格式 CSV。

没有真实数据时，可先用合成数据跑通全流程：

```bash
uv run emoji synth-data --out data/synth/smoke --samples-per-class 24 --seed 0
```

生成 `data/synth/smoke/fer2013.csv`（与 FER2013 同格式）与 `meta.json`。

## 快速开始

### 1) 合成数据冒烟（无需下载数据，CPU 约 1 分钟）

```bash
uv run emoji synth-data --out data/synth/smoke --samples-per-class 24 --seed 0
uv run emoji train --config configs/smoke.yaml
uv run emoji eval --checkpoint outputs/smoke/best.pt --split PrivateTest
```

`train` 会在 `outputs/smoke/` 下生成：

| 文件 | 说明 |
|---|---|
| `best.pt` / `last.pt` | 最优 / 最近 checkpoint（last 额外含 optimizer、scheduler 状态） |
| `history.csv` | 每个 epoch 的 lr、train/val loss、acc、macro-F1、耗时 |
| `metrics.json` | 训练摘要（best/final 验证指标、参数量、数据划分、耗时等） |
| `config.yaml` | 本次实验的 resolved config（复现必需） |
| `command.txt` | 实际执行的命令行 |

`eval` 在 `outputs/smoke/eval_PrivateTest/` 下生成 `metrics.json`、`per_class.csv`、
`confusion_matrix.png`，并打印 batch=1 的推理时延与 FPS。

> 也可以一条命令完成冒烟（缺少合成数据时自动生成）：
> `uv run emoji train --smoke`
>
> 注意：冒烟只有约 15 个训练 batch，`configs/smoke.yaml` 关闭了数据增强
> （`data.augment: false`）并通过 `model.bn_warmup_batches: 30` 先校准 BN 统计量；
> 合成的 7 类数据仅用于验证「数据→训练→评估」链路能跑通，其准确率不代表真实
> FER 性能（正式实验请使用 `configs/base.yaml` 系列）。

### 2) 真实 FER2013 训练

```bash
# 基线：ResNet18（48×48 输入使用改造 stem，见“实现要点”）
uv run emoji train --config configs/baseline_resnet18.yaml

# 改进模型：ResNet18 + CBAM
uv run emoji train --config configs/cbam_resnet18.yaml

# 轻量模型：MobileNetV3-small（实时演示候选）
uv run emoji train --config configs/mobilenetv3_small.yaml
```

输出目录名由配置里的 `run_name` 决定（`outputs/<run_name>/`）；同名目录已存在时会自动
追加时间后缀，不会覆盖历史结果。常用覆盖示例：

```bash
# 换随机种子 / 换损失函数 / 开 CBAM / 关数据增强
uv run emoji train --config configs/baseline_resnet18.yaml \
  --set seed=123 --set run_name=baseline-s123 \
  --set train.loss=focal --set train.class_weight=balanced \
  --set model.use_cbam=true --set data.augment=false
```

### 3) 评估

```bash
# 默认使用 checkpoint 里 resolved config 的 eval.split（FER2013 为 PrivateTest）
uv run emoji eval --checkpoint outputs/cbam-resnet18/best.pt

# 指定划分与输出目录
uv run emoji eval --checkpoint outputs/cbam-resnet18/best.pt \
  --split PublicTest --out outputs/cbam-resnet18/eval_public
```

### 4) 鲁棒性实验

```bash
uv run emoji robustness --checkpoint outputs/cbam-resnet18/best.pt \
  --split PrivateTest --severities 1,2,3,4,5
```

输出 `outputs/cbam-resnet18/robustness_PrivateTest/robustness.csv`，列含义：

- `perturbation`：`clean` / `gaussian_noise` / `random_erasing` / `downscale`；
- `severity`：扰动强度 1–5（`clean` 为 0）；
- `acc`、`macro_f1`：该扰动下的指标；
- `acc_drop`、`macro_f1_drop`：相对干净基线的**绝对下降（百分点）**。

### 5) Grad-CAM 可解释性分析

```bash
uv run emoji gradcam --checkpoint outputs/cbam-resnet18/best.pt \
  --split PrivateTest --top-k 8
```

输出 `outputs/cbam-resnet18/gradcam_PrivateTest/`：`wrong_*.png`、`correct_*.png` 与
`gradcam_grid.png` 总览图；默认对预测类别生成 CAM，标题中同时标注真实/预测标签。

### 6) 摄像头实时演示

```bash
uv sync --extra demo
uv run emoji demo --checkpoint outputs/mobilenet-v3-small/best.pt --camera 0
```

- 检测器优先 MediaPipe，失败自动回退 OpenCV Haar；两者都缺失时给出安装提示；
- 界面显示类别、置信度、FPS 与最近 30 帧的表情统计；按 `q` 或 `ESC` 退出；
- 摄像头不可用（服务器/容器）时只会在启动时报错，核心训练评估流程不受影响。

### 7) 测试与代码检查

```bash
uv run pytest -q              # 全部 CPU、合成数据，总计 < 90s
uv run ruff check src tests   # 代码风格检查（line-length=100）
```

### 8) 重新生成开题报告 DOCX / PDF

Markdown 是唯一正文源文件，docx/pdf 由脚本重建（需要 LibreOffice `soffice` 与
poppler 的 `pdfinfo/pdfimages`，Ubuntu 上 `apt install libreoffice poppler-utils`）：

```bash
python3 tools/build_doc.py          # doc/xxx.md -> doc/xxx.docx + doc/xxx.pdf
python3 tools/make_tech_route.py    # 可选：重新生成 doc/tech_route_fer.png
```

实现方式：Markdown → HTML（CSS 复刻原排版，图片 base64 内嵌）→ LibreOffice
html→odt→docx（套用 `tools/docx_style_template.xml` 中的原文档样式表：A4、
宋体正文 12pt、黑体标题、表格边框与列宽）→ docx→pdf；构建时会校验 docx 内嵌
图片、表格数量与 pdf 页数。

## 配置系统

- 配置为 YAML，支持 `_base_` 继承（相对当前文件路径）：

  ```yaml
  _base_: [base.yaml]
  model: {name: resnet18_fer, use_cbam: true}
  ```

- 命令行 `--set a.b=value` 覆盖任意字段，值按 YAML 解析（保留 int/float/bool 类型），
  例如 `--set train.lr=1e-3 --set model.use_cbam=true --set data.class_names=[a,b]`；
- 未知字段会直接报错（防止拼写错误被静默忽略）；
- `emoji eval/robustness/gradcam --set ...` 的覆盖项**只应用于数据/评估字段**，
  模型结构以 checkpoint 中保存的配置为准，避免权重与结构不匹配。

## 实验计划矩阵（对应开题报告第 4/5 章与附录 A）

| 实验 | 命令 / 配置 | 主要指标 | 状态 |
|---|---|---|---|
| 主对比：ResNet18 / MobileNetV3 / CBAM-ResNet | `configs/baseline_resnet18.yaml`、`configs/cbam_resnet18.yaml`、`configs/mobilenetv3_small.yaml` | Acc、Macro-F1、参数量、时延 | 待真实数据 |
| 消融：CBAM / 数据增强 / 损失函数 / 输入尺寸 | `--set model.use_cbam=false`、`--set data.augment=false`、`--set train.loss=ce\|lsce\|focal\|cbce`、`--set data.image_size=96` | Acc、Macro-F1 | 待真实数据 |
| 类别不平衡分析 | `--set train.class_weight=balanced\|effective`、`--set train.loss=cbce` | per-class P/R/F1、混淆矩阵 | 待真实数据 |
| 跨数据集（FER2013→RAF-DB / CK+） | 先转换 CSV，再 `--set data.csv_path=<跨域csv> --set data.val_split=...` | Acc、Macro-F1 变化 | 待数据适配（TODO） |
| 鲁棒性（噪声/遮挡/降分辨率） | `emoji robustness ...` | 性能下降曲线 | 已实现，待真实数据 |
| 错误分析 | `emoji eval` 的 per_class.csv + `emoji gradcam` | 定性 + 定量 | 已实现 |
| 系统性能 | `emoji demo` + `emoji eval` 时延统计 | FPS、端到端时延 | 已实现（需摄像头/GPU 实测） |

## 实现要点（与开题报告方案的对应）

1. **48×48 输入适配**：ResNet18 在 `input_size<=64` 时使用 3×3/stride1 卷积并去掉
   maxpool，使末端特征为 6×6（标准 stem 会退化到 1×1）；96/224 输入使用标准 ImageNet
   stem。灰度复制为 3 通道可复用 ImageNet 权重（`data.gray_to_rgb`）。
2. **CBAM 插入位置（标准做法）**：ResNet18 在**每个 BasicBlock 内部** ``conv2/bn2``
   之后、残差相加之前应用 CBAM；MobileNetV3 在 ``InvertedResidual`` 的卷积分支末端
   （残差相加之前）应用，无残差块（如 stem）则接在卷积输出之后。两者都不改变特征图
   形状，便于与基线公平对比。
3. **数据划分**：训练固定 `Training`，验证用 `PublicTest`，`PrivateTest` 只用于最终
   评估；`--set data.val_split` 可改验证划分（如跨库实验）。
4. **训练细节**：AdamW/SGD、warmup + 余弦退火、梯度裁剪、早停、AMP（CPU 自动关闭）、
   3–5 seeds 可复现；每 epoch 一行日志与 CSV。可选 `model.bn_warmup_batches`
   在训练前校准 BN running stats（冒烟配置启用，避免小数据短训练时 eval 指标异常）。
5. **鲁棒性扰动在标准化空间施加**（干净图均值≈0、方差≈1）：噪声 sigma 相对标准差，
   擦除用 0（数据均值）填充，降分辨率用双线性下采样再上采样。
6. **实时演示**：`cv2`/`mediapipe` 延迟导入；核心训练评估流程不依赖它们。

## 复现与记录规范

- 每次实验的 `outputs/<run>/config.yaml` + `history.csv` + `metrics.json` 一并保留，
  写论文时可直接引用；**不要手工 cherry-pick 最好的一次**，报告 3–5 seeds 的 mean±std；
- 对比实验固定随机种子集合（例如 `--set seed=0/1/2`），并固定硬件/线程数报告时延
  （`emoji eval` 打印的是 batch=1 中位时延与 FPS）；
- `PrivateTest` 只用于最终报告一次，模型选择/早停只用 `PublicTest`；
- 数据与输出目录（`data/raw`、`data/synth`、`outputs/`）均已 gitignore，不要提交；
- 提交代码前运行 `uv run pytest -q` 与 `uv run ruff check src tests`。

## 已知限制 / TODO

- [ ] 真实 FER2013 实验尚未跑（基线精度待补，预计 ResNet18 约 71–73%）。
- [ ] 跨数据集：目前只支持 FER2013 CSV 格式；RAF-DB / CK+ 需先转 CSV，
      且 CK+ 的 7 类含 contempt、无 neutral，需显式类别映射（TODO：原生 Dataset）。
- [ ] FERPlus 软标签 / 多标签训练尚未实现（开题报告中的可选项）。
- [ ] 多 seed 批量脚本与结果汇总表（`scripts/run_matrix.sh`、`results.csv`）待补。
- [ ] `data.augment=false` 可直接用于数据增强消融，但尚未提供"仅旋转/仅翻转"等细粒度开关。
- [ ] 实时演示未在真实摄像头环境验证；MediaPipe 对 Python 版本较敏感，失败会自动回退 Haar。
- [ ] 跨数据集微调（先在目标域训练若干 epoch 再测）尚未封装成命令，当前可手动换 CSV。
- [ ] 推理时延为 CPU 测量（当前开发机无可用 GPU），论文中需注明硬件与线程配置。

## 参考文献

完整参考文献（共 16 篇，含 SCN、TransFER、POSTER 等 2021 年后工作）见开题报告
[`doc/开题报告_基于深度学习的人脸表情识别算法研究.md`](<doc/开题报告_基于深度学习的人脸表情识别算法研究.md>)
的「参考文献」一节，代码与文档保持同一套编号，此处不再重复。

## 许可

代码用于本科毕业设计研究；数据集遵循各自原始许可（见 [`data/README.md`](data/README.md)）。
