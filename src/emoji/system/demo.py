"""摄像头实时人脸表情识别演示。

依赖策略：``cv2`` 与 ``mediapipe`` 都是**延迟导入**的，核心训练/评估流程不依赖它们。
安装演示依赖：``uv sync --extra demo``。

检测器优先级：MediaPipe Face Detection → OpenCV Haar 级联；两者都不可用时给出
明确的安装提示。系统为 "人脸检测 → 表情分类 → 可视化" 流程，界面显示类别、
置信度、FPS 与最近 N 帧的表情统计。
"""

from __future__ import annotations

import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

# 终端打印用的中文对照表（界面上仍用英文标签，避免 OpenCV 中文字体问题）
CLASS_NAMES_ZH: dict[str, str] = {
    "angry": "愤怒",
    "disgust": "厌恶",
    "fear": "恐惧",
    "happy": "高兴",
    "sad": "悲伤",
    "surprise": "惊讶",
    "neutral": "中性",
}


def _import_cv2() -> Any:
    """延迟导入 OpenCV，缺失时给出安装提示。"""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - 由环境决定
        raise RuntimeError(
            "实时演示需要 opencv-python，请先安装演示依赖：\n"
            "  uv sync --extra demo\n"
            "或:\n"
            "  uv pip install opencv-python mediapipe"
        ) from exc
    return cv2


def crop_face(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    margin: float = 0.25,
) -> np.ndarray:
    """按人脸框（可外扩 margin）裁剪，并保证不越界、返回非空数组。"""
    if frame.ndim not in (2, 3):
        raise ValueError(f"frame 维度异常: {frame.shape}")
    if margin < 0:
        raise ValueError("margin 不能为负")
    height, width = frame.shape[:2]
    x, y, box_w, box_h = (int(v) for v in box)
    margin_x = int(box_w * margin)
    margin_y = int(box_h * margin)
    x0 = max(0, x - margin_x)
    y0 = max(0, y - margin_y)
    x1 = min(width, x + box_w + margin_x)
    y1 = min(height, y + box_h + margin_y)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"人脸框无效: box={box}, frame={frame.shape}")
    return frame[y0:y1, x0:x1]


class FaceDetector:
    """人脸检测器：优先 MediaPipe，失败时回退 OpenCV Haar 级联。"""

    def __init__(self, prefer_mediapipe: bool = True) -> None:
        self._cv2 = _import_cv2()
        self._mediapipe_detector: Any = None
        self._haar: Any = None
        self.backend = ""
        if prefer_mediapipe:
            try:
                import mediapipe as mp

                self._mediapipe_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=0,
                    min_detection_confidence=0.5,
                )
                self.backend = "mediapipe"
                return
            # MediaPipe 未安装 / 版本不兼容 / 初始化失败都应该回退到 Haar
            except Exception as exc:  # noqa: BLE001
                print(f"[demo] MediaPipe 不可用（{exc}），回退 OpenCV Haar 检测器")
        cascade_path = self._cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._haar = self._cv2.CascadeClassifier(cascade_path)
        if self._haar.empty():
            raise RuntimeError(
                "OpenCV Haar 级联加载失败；请检查 opencv-python 安装是否完整，"
                "或安装 MediaPipe：uv sync --extra demo"
            )
        self.backend = "haar"

    def detect(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        """返回画面中所有人脸的 ``(x, y, w, h)`` 像素坐标框。"""
        if self.backend == "mediapipe":
            return self._detect_mediapipe(frame_bgr)
        return self._detect_haar(frame_bgr)

    def _detect_mediapipe(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        cv2 = self._cv2
        height, width = frame_bgr.shape[:2]
        result = self._mediapipe_detector.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        boxes: list[tuple[int, int, int, int]] = []
        if not result.detections:
            return boxes
        for detection in result.detections:
            relative = detection.location_data.relative_bounding_box
            x = max(0, int(relative.xmin * width))
            y = max(0, int(relative.ymin * height))
            box_w = min(width - x, int(relative.width * width))
            box_h = min(height - y, int(relative.height * height))
            if box_w > 0 and box_h > 0:
                boxes.append((x, y, box_w, box_h))
        return boxes

    def _detect_haar(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        cv2 = self._cv2
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._haar.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    def close(self) -> None:
        """释放 MediaPipe 资源（Haar 无需释放）。"""
        if self._mediapipe_detector is not None:
            self._mediapipe_detector.close()
            self._mediapipe_detector = None


def _draw_hud(
    cv2: Any,
    frame: np.ndarray,
    label: str | None,
    confidence: float,
    fps: float,
    recent: deque[str],
) -> None:
    """在画面上绘制类别、置信度、FPS 与近期统计。"""
    lines = [f"FPS: {fps:5.1f}"]
    if label is not None:
        lines.append(f"{label.upper()}  {confidence * 100:5.1f}%")
    if recent:
        counter = Counter(recent)
        total = sum(counter.values())
        stats = "  ".join(
            f"{name}:{count * 100 // total}%" for name, count in counter.most_common(3)
        )
        lines.append(f"last {total}: {stats}")
    for index, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (10, 24 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def run_demo(
    ckpt_path: str | Path,
    camera: int = 0,
    width: int = 640,
    height: int = 480,
    window: int = 30,
    prefer_mediapipe: bool = True,
    device: str | None = None,
) -> None:
    """打开摄像头实时演示表情识别，按 ``q`` 或 ESC 退出。"""
    from emoji.data.transforms import EvalTransform
    from emoji.evaluate import load_model_from_checkpoint

    cv2 = _import_cv2()
    model, cfg, _ckpt = load_model_from_checkpoint(ckpt_path, device=device)
    device_obj = next(model.parameters()).device
    class_names = list(cfg.data.class_names)[: cfg.data.num_classes]
    transform = EvalTransform(
        out_chans=cfg.data.model_in_chans,
        mean=cfg.data.mean,
        std=cfg.data.std,
    )
    detector = FaceDetector(prefer_mediapipe=prefer_mediapipe)
    print(f"[demo] 检测器={detector.backend}，模型={cfg.model.name}，设备={device_obj}")
    print("[demo] 中文对照：" + "，".join(f"{en}={zh}" for en, zh in CLASS_NAMES_ZH.items()))
    print("[demo] 按 q 或 ESC 退出")

    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        detector.close()
        raise RuntimeError(
            f"无法打开摄像头 {camera}；请确认设备存在、没有被其它程序占用，"
            "或换一个 --camera 编号（如 1）"
        )
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    recent: deque[str] = deque(maxlen=max(1, window))
    fps_ema = 0.0
    last_time = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("摄像头读取失败（设备可能已断开）")
            boxes = detector.detect(frame)
            label: str | None = None
            confidence = 0.0
            if boxes:
                # 取面积最大的人脸（多人场景下优先离镜头最近的人）
                box = max(boxes, key=lambda item: item[2] * item[3])
                face_bgr = crop_face(frame, box)
                gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(
                    gray,
                    (cfg.data.image_size, cfg.data.image_size),
                    interpolation=cv2.INTER_AREA,
                )
                tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).unsqueeze(0)
                inputs = transform(tensor).unsqueeze(0).to(device_obj)
                with torch.no_grad():
                    probs = torch.softmax(model(inputs), dim=1)[0]
                confidence = float(probs.max().item())
                label = class_names[int(probs.argmax().item())]
                recent.append(label)
                x, y, box_w, box_h = box
                cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)

            now = time.perf_counter()
            elapsed = now - last_time
            last_time = now
            if elapsed > 0:
                instant = 1.0 / elapsed
                fps_ema = instant if fps_ema == 0 else 0.9 * fps_ema + 0.1 * instant
            _draw_hud(cv2, frame, label, confidence, fps_ema, recent)
            cv2.imshow("FER demo - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        capture.release()
        detector.close()
        cv2.destroyAllWindows()
