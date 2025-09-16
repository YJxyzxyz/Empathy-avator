# -*- coding: utf-8 -*-
"""Rule-based fusion of multimodal emotion probabilities."""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np


class RuleFusion:
    """Simple weighted fusion with gating based on per-modality confidence."""

    def __init__(
        self,
        classes: Sequence[str],
        base_weights: Optional[Mapping[str, float]] = None,
        min_confidence: float = 0.15,
    ) -> None:
        self.classes: Sequence[str] = tuple(classes)
        if base_weights is None:
            base_weights = {"vision": 0.5, "audio": 0.3, "text": 0.2}
        self.base_weights = dict(base_weights)
        self.min_confidence = float(min_confidence)

    def _validate_probs(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=np.float32)
        if probs.size != len(self.classes):
            raise ValueError("probabilities length mismatch")
        total = float(np.sum(probs))
        if total <= 0:
            probs = np.ones(len(self.classes), dtype=np.float32)
        else:
            probs = probs / total
        return probs

    def _gated_weight(self, name: str, conf: Optional[float]) -> float:
        base = float(self.base_weights.get(name, 0.0))
        if conf is None:
            return 0.0
        conf = float(conf)
        if conf <= 0:
            return 0.0
        if conf < self.min_confidence:
            return base * (conf / max(self.min_confidence, 1e-6))
        return base * min(1.5, 0.6 + conf)

    def fuse(self, modalities: Mapping[str, Mapping[str, object]]) -> Dict[str, object]:
        weighted = np.zeros(len(self.classes), dtype=np.float32)
        used_weights: Dict[str, float] = {}
        per_modality = {}

        for name, info in modalities.items():
            probs = info.get("probs") if info else None
            if probs is None:
                continue
            try:
                probs_arr = self._validate_probs(np.asarray(probs))
            except Exception:
                continue
            conf = float(info.get("conf", float(np.max(probs_arr)))) if info else 0.0
            w = self._gated_weight(name, conf)
            if w <= 0:
                continue
            weighted += probs_arr * w
            used_weights[name] = w
            top_idx = int(np.argmax(probs_arr))
            per_modality[name] = {
                "emo": self.classes[top_idx],
                "conf": conf,
                "weight": w,
                "probs": probs_arr.tolist(),
            }

        total_w = float(sum(used_weights.values()))
        if total_w <= 0:
            fused = np.zeros(len(self.classes), dtype=np.float32)
            fused[self.classes.index("neutral")] = 1.0
        else:
            fused = weighted / total_w

        max_idx = int(np.argmax(fused))
        fused_conf = float(np.max(fused))

        return {
            "probs": fused.tolist(),
            "emo": self.classes[max_idx],
            "conf": fused_conf,
            "weights": used_weights,
            "per_modality": per_modality,
        }
