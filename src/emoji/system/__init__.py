"""实时演示子包（cv2 / mediapipe 依赖延迟导入）。"""

from __future__ import annotations

from emoji.system.demo import CLASS_NAMES_ZH, FaceDetector, crop_face, run_demo

__all__ = ["CLASS_NAMES_ZH", "FaceDetector", "crop_face", "run_demo"]
