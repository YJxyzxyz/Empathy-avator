# models/fer/infer.py
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import onnxruntime as ort

try:
    import mediapipe as mp
    MP_OK = True
except Exception:
    MP_OK = False

DEFAULT_CLASSES = ('angry','disgust','fear','happy','sad','surprise','neutral')

class FER:
    """
    轻量 FER 推理：
    - 人脸检测：MediaPipe（可选，没装就用全图）
    - 模型：ONNX（NCHW, 224x224, 归一化 (x-0.5)/0.5）
    返回：
      {"probs": np.ndarray(C), "conf": float, "bbox": (x1,y1,x2,y2) 或 None}
    """
    def __init__(self, onnx_path: str, input_size=(224,224), classes=DEFAULT_CLASSES):
        self.onnx_path = onnx_path
        self.input_size = tuple(input_size)
        self.classes = tuple(classes)

        # 尝试优先用 CUDA，失败则回落到 CPU
        providers = []
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(self.onnx_path, providers=providers)
        except Exception:
            self.session = ort.InferenceSession(self.onnx_path, providers=['CPUExecutionProvider'])

        # MediaPipe 人脸检测（可选）
        self.face = None
        if MP_OK:
            self.face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

        # 记录输入名
        self.input_name = self.session.get_inputs()[0].name

    def _detect_face(self, frame_bgr):
        """
        返回裁剪框 (x1,y1,x2,y2)。若无检测器或未检出，返回 None。
        """
        if self.face is None or frame_bgr is None:
            return None
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        det = self.face.process(rgb)
        if not det or not det.detections:
            return None
        # 取置信度最高的人脸
        box = max(det.detections, key=lambda d: d.score[0])
        rel = box.location_data.relative_bounding_box
        x1, y1 = int(rel.xmin*w), int(rel.ymin*h)
        x2, y2 = x1 + int(rel.width*w), y1 + int(rel.height*h)
        # 边界裁剪
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def _preprocess(self, bgr):
        img = cv2.resize(bgr, self.input_size).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = np.transpose(img, (2, 0, 1))[None, ...]  # NCHW
        return img

    def infer(self, frame_bgr):
        if frame_bgr is None or frame_bgr.size == 0:
            # 兜底：均匀分布 + 低置信度
            c = len(self.classes)
            return {"probs": np.ones(c, dtype=np.float32) / c, "conf": 0.2, "bbox": None}

        bbox = self._detect_face(frame_bgr)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            crop = frame_bgr[y1:y2, x1:x2]
        else:
            crop = frame_bgr  # 没检测器或没检出，就用整图
            bbox = None

        inp = self._preprocess(crop)
        logits = self.session.run(None, {self.input_name: inp})[0][0]  # (C,)

        # softmax
        e = np.exp(logits - np.max(logits))
        probs = e / np.sum(e)
        conf = float(np.max(probs))
        return {"probs": probs, "conf": conf, "bbox": bbox}
