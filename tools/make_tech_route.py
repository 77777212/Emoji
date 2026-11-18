#!/usr/bin/env python3
"""生成开题报告技术路线图 doc/tech_route_fer.png（2024x1264，白底）。

用法：python3 tools/make_tech_route.py  （依赖 matplotlib、Pillow、系统中文字体）
"""
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = str(Path(__file__).resolve().parents[1])
FONT_REG = "/usr/share/fonts/adobe-source-han-sans/SourceHanSansCN-Regular.otf"
FONT_MED = "/usr/share/fonts/adobe-source-han-sans/SourceHanSansCN-Medium.otf"
FONT_BOLD = "/usr/share/fonts/adobe-source-han-sans/SourceHanSansCN-Bold.otf"
for p in (FONT_REG, FONT_MED, FONT_BOLD):
    font_manager.fontManager.addfont(p)

W, H = 2024, 1264
DPI = 200
fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

TITLE = FontProperties(fname=FONT_BOLD, size=16)
SUB = FontProperties(fname=FONT_REG, size=11)
SIDE = FontProperties(fname=FONT_REG, size=11.5)

BLUE_F, BLUE_E = "#EAF2FB", "#6E9BD1"
GREEN_F, GREEN_E = "#EAF6EC", "#5FAE78"
ORANGE_F, ORANGE_E = "#FDF3E3", "#D9A84E"
TXT = "#1F2D3D"
TXT_SUB = "#33475B"
SIDE_C = "#7C8A99"
ARROW_C = "#5B6B7F"


def rounded_box(x0, y0, w, h, face, edge):
    patch = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0,rounding_size=18",
        linewidth=2.2, edgecolor=edge, facecolor=face, zorder=2,
        mutation_aspect=1.0,
    )
    ax.add_patch(patch)


def center_text(x, y, s, fp, color=TXT, **kwargs):
    ax.text(x, y, s, ha="center", va="center", fontproperties=fp,
            color=color, zorder=4, **kwargs)


def arrow(x0, y0, x1, y1):
    a = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", mutation_scale=24,
        linewidth=2.4, color=ARROW_C,
        shrinkA=0, shrinkB=0,
        connectionstyle="arc3,rad=0",
        zorder=3,
    )
    ax.add_patch(a)


# ---------- 五个阶段框 ----------
L1 = (300, 1062, 1470, 160)
L2A = (270, 802, 680, 160)
L2B = (1120, 802, 680, 160)
L3 = (300, 547, 1470, 160)
L4 = (300, 292, 1470, 160)
L5 = (300, 42, 1470, 160)

rounded_box(*L1, BLUE_F, BLUE_E)
rounded_box(*L2A, BLUE_F, BLUE_E)
rounded_box(*L2B, BLUE_F, BLUE_E)
rounded_box(*L3, GREEN_F, GREEN_E)
rounded_box(*L4, GREEN_F, GREEN_E)
rounded_box(*L5, ORANGE_F, ORANGE_E)

# ---------- 阶段一：数据准备 ----------
center_text(1035, 1165, "数据准备", TITLE)
center_text(1035, 1098,
            "FER2013 / RAF-DB / CK+ · 官方划分 · 灰度归一化（48×48 / 96×96 两档） · 训练集数据增强",
            SUB, TXT_SUB)

# ---------- 阶段二：基线复现 / 改进模型 ----------
center_text(610, 910, "基线复现", TITLE)
center_text(610, 866, "ResNet18 / MobileNetV3-small/large", SUB, TXT_SUB)
center_text(610, 834, "ImageNet 预训练 · stem 适配 · 48/96 输入", SUB, TXT_SUB)

center_text(1460, 910, "改进模型", TITLE)
center_text(1460, 866, "CBAM 注意力 + 数据增强", SUB, TXT_SUB)
center_text(1460, 834, "损失对比：CE / 标签平滑 / 加权 / Focal", SUB, TXT_SUB)

# ---------- 阶段三：实验验证 ----------
center_text(1035, 650, "实验验证", TITLE)
center_text(1035, 590,
            "主对比 · 消融 · 类别不平衡 · 跨域迁移（RAF-DB） · 鲁棒性 · Grad-CAM",
            SUB, TXT_SUB)

# ---------- 阶段四：原型系统 ----------
center_text(1035, 395, "原型系统", TITLE)
center_text(1035, 337,
            "MediaPipe / OpenCV 人脸检测 + 模型推理 + 可视化 · 目标 ≥ 15 FPS",
            SUB, TXT_SUB)

# ---------- 阶段五：成果输出 ----------
center_text(1035, 145, "成果输出", TITLE)
center_text(1035, 87, "学位论文 · 开源代码仓库 · 演示视频", SUB, TXT_SUB)

# ---------- 箭头 ----------
arrow(1035, 1062, 610, 962)    # 数据准备 -> 基线复现
arrow(950, 882, 1120, 882)     # 基线复现 -> 改进模型
arrow(1460, 802, 1035, 707)    # 改进模型 -> 实验验证
arrow(1035, 547, 1035, 452)    # 实验验证 -> 原型系统
arrow(1035, 292, 1035, 202)    # 原型系统 -> 成果输出

# ---------- 左侧阶段标注 ----------
ax.text(150, 882, "研\n究\n阶\n段", ha="center", va="center",
        fontproperties=SIDE, color=SIDE_C, linespacing=1.6, zorder=4)
ax.text(150, 372, "应\n用\n阶\n段", ha="center", va="center",
        fontproperties=SIDE, color=SIDE_C, linespacing=1.6, zorder=4)

out = os.path.join(BASE, "doc", "tech_route_fer.png")
fig.savefig(out, dpi=DPI, facecolor="white")
plt.close(fig)

from PIL import Image
im = Image.open(out)
print("saved:", out)
print("size:", im.size, "mode:", im.mode)
print("bytes:", os.path.getsize(out))
