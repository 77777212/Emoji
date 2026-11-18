# 数据准备说明

本仓库**不提交任何数据文件**（见根目录 `.gitignore`）。所有数据放在 `data/raw/` 下，
合成数据由命令生成到 `data/synth/`（同样被 gitignore）。

## 目录布局

```text
data/
├── README.md              # 本文件
├── raw/                   # 真实数据集（不入库）
│   ├── fer2013/
│   │   └── fer2013.csv    # Kaggle FER2013 原始 CSV（必需）
│   ├── rafdb/             # RAF-DB（跨数据集实验，可选）
│   └── ckplus/            # CK+（跨数据集实验，可选）
└── synth/                 # 合成数据（命令生成，不入库）
    └── smoke/
        ├── fer2013.csv    # 与 FER2013 同格式
        └── meta.json      # 生成参数与各划分样本数
```

## 1. FER2013（主数据集，必需）

- 来源：Kaggle 竞赛 *Challenges in Representation Learning: Facial Expression
  Recognition Challenge*（Goodfellow et al., 2013）。
- 获取：登录 Kaggle → 搜索 "FER2013" → 下载 `fer2013.csv` →
  放到 `data/raw/fer2013/fer2013.csv`。
- 格式：`emotion,pixels,Usage`
  - `emotion`：0..6 → angry / disgust / fear / happy / sad / surprise / neutral；
  - `pixels`：48×48=2304 个空格分隔的灰度像素（0..255）；
  - `Usage`：`Training` / `PublicTest` / `PrivateTest`。
- 官方划分（35,887 张）：Training 28,709 / PublicTest 3,589 / PrivateTest 3,589。
  **PublicTest 只用于验证/早停，PrivateTest 只用于最终报告一次**，避免模型选择泄漏。
- 使用前请阅读 Kaggle 页面上的数据条款：图片来自互联网，仅供研究使用。

## 2. RAF-DB（跨数据集实验，建议）

- 来源：Li & Deng, CVPR 2017，官网申请：<http://www.whdeng.cn/RAF/model1.html>
- 用途：验证真实场景（in-the-wild）跨域泛化，弥补 CK+ 过于理想化的问题。
- 注意：Basic 7 类与 FER2013 的 7 类标签空间一致（含 neutral），适合做
  FER2013→RAF-DB 的直接迁移 / 微调对比；官方提供 train / test 划分。
- 当前代码只原生支持 FER2013 CSV；使用 RAF-DB 前需先转换为同格式 CSV
  （见下方"跨数据集 CSV 转换"）。

## 3. CK+（补充跨数据集实验，可选）

- 来源：Lucey et al., CVPRW 2010，需邮件/表单申请。
- 规模：593 个序列，其中 327 个带表情标签（常用 981 张峰值帧）。
- 注意：**标签空间与 FER2013 不一致**——CK+ 标准 7 类含 contempt（轻蔑）且无
  neutral。若做跨库评估必须明确类别映射（常见做法：丢弃 contempt，只用 6 类公共
  类别），并在论文中说明，否则跨库数字不可比。
- 图像为 640×480 左右彩图，需先人脸检测/对齐/灰度化再缩放到 48×48。

## 4. 合成数据（无真实数据时的冒烟/测试用）

```bash
uv run emoji synth-data --out data/synth/smoke --samples-per-class 24 --seed 0
```

生成与 FER2013 完全同构的 CSV（每类不同位置的亮条 + 高斯噪声，类别间可分），
默认按 70%/15%/15% 划分 Training/PublicTest/PrivateTest。仅用于跑通流程与 pytest，
**不能用于汇报实验精度**。

## 跨数据集 CSV 转换（RAF-DB / CK+）

当前 Dataset 只读取 FER2013 CSV。把 RAF-DB 的 basic 7 类图片转成同格式即可直接复用
训练/评估流程（示例思路，待补充正式转换脚本）：

1. 读取官方标注（RAF-DB 为 `label` 列表文件）；
2. 人脸区域裁剪后转灰度，resize 到 48×48（保存时不要做归一化）；
3. 像素展平成 2304 个 0..255 的整数，按 `emotion,pixels,Usage` 写成 CSV；
4. `Usage` 用 ` Training` 之外的取值区分来源划分，例如 RAF-DB 的 train → `Training`，
   test → `PrivateTest`；`--set data.csv_path=<新csv>` 即可训练/评估。

> TODO：后续可增加 `RAFDBDataset` / `CKPlusDataset` 原生读取与类别映射表。

## 许可与合规提醒

- FER2013 图片版权归原始来源所有，仅用于学术研究；不要重新分发数据集。
- RAF-DB、CK+ 均为申请制，使用需遵守各自协议；论文/报告中引用对应文献。
- 本仓库的代码只做数据读取与实验，不包含任何数据集内容。
