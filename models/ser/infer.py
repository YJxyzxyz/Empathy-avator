# -*- coding: utf-8 -*-
"""Simple streaming Speech Emotion Recognition (SER).

This module implements a lightweight, rule-based SER engine that analyses
incoming audio chunks and produces emotion probabilities for the same set of
classes used by FER.  It is intentionally dependency-light so the demo can run
on edge devices without heavy models.  The logic is heuristic but the interface
matches what a learned model would expose, which makes it easy to swap later.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

DEFAULT_CLASSES: Sequence[str] = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
)


@dataclass
class SERState:
    """Holds the latest SER inference results."""

    probs: np.ndarray
    conf: float
    features: Dict[str, float]
    raw_probs: np.ndarray

    def to_dict(self) -> Dict[str, object]:
        return {
            "probs": self.probs.tolist(),
            "conf": float(self.conf),
            "features": {k: float(v) for k, v in self.features.items()},
        }


class StreamingSER:
    """A very small SER module with heuristic scoring.

    Parameters
    ----------
    sample_rate: int
        Target sampling rate for analysis.  Incoming audio is linearly
        resampled if needed.
    window_seconds: float
        Audio window length used for feature extraction.  Short windows make the
        system responsive while longer windows are more stable.
    smooth: int
        Number of recent probability vectors to average for temporal smoothing.
    classes: Sequence[str]
        Emotion classes (aligned with FER).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        window_seconds: float = 1.2,
        smooth: int = 4,
        classes: Sequence[str] = DEFAULT_CLASSES,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.window_seconds = float(window_seconds)
        self.window_samples = max(1, int(self.sample_rate * self.window_seconds))
        self.classes: Sequence[str] = tuple(classes)
        self._buffer: np.ndarray = np.zeros(0, dtype=np.float32)
        self._prob_hist: Deque[np.ndarray] = collections.deque(maxlen=max(1, int(smooth)))
        self._state: Optional[SERState] = None

    # ------------------------------------------------------------------ utils
    def _ensure_float(self, audio: np.ndarray) -> np.ndarray:
        if audio.dtype == np.float32 or audio.dtype == np.float64:
            return audio.astype(np.float32, copy=False)
        if np.issubdtype(audio.dtype, np.integer):
            info = np.iinfo(audio.dtype)
            audio = audio.astype(np.float32) / max(1.0, float(info.max))
            return audio
        return audio.astype(np.float32)

    def _linear_resample(self, audio: np.ndarray, src_sr: int) -> np.ndarray:
        if src_sr == self.sample_rate or audio.size == 0:
            return audio
        ratio = float(self.sample_rate) / float(src_sr)
        tgt_n = max(1, int(round(audio.size * ratio)))
        x_old = np.linspace(0.0, 1.0, audio.size, dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, tgt_n, dtype=np.float32)
        return np.interp(x_new, x_old, audio).astype(np.float32)

    def _update_buffer(self, audio: np.ndarray) -> None:
        if audio.size == 0:
            return
        if self._buffer.size == 0:
            self._buffer = audio
        else:
            self._buffer = np.concatenate([self._buffer, audio])
        if self._buffer.size > self.window_samples:
            self._buffer = self._buffer[-self.window_samples :]

    def _extract_window(self) -> np.ndarray:
        if self._buffer.size < self.window_samples // 4:
            return np.zeros(0, dtype=np.float32)
        if self._buffer.size < self.window_samples:
            pad = np.zeros(self.window_samples - self._buffer.size, dtype=np.float32)
            return np.concatenate([pad, self._buffer])
        return self._buffer.copy()

    def _compute_features(self, audio: np.ndarray) -> Dict[str, float]:
        if audio.size == 0:
            return {"rms": 0.0, "zcr": 0.0, "centroid": 0.0}
        audio = audio - float(np.mean(audio))
        rms = float(np.sqrt(np.mean(np.square(audio))))
        # zero crossing rate
        signs = np.sign(audio)
        signs[signs == 0] = 1
        zcr = float(np.mean(signs[:-1] != signs[1:]))
        # spectral centroid
        spectrum = np.abs(np.fft.rfft(audio))
        if spectrum.size == 0:
            centroid = 0.0
        else:
            freqs = np.fft.rfftfreq(audio.size, d=1.0 / self.sample_rate)
            total = float(np.sum(spectrum))
            centroid = float(np.sum(freqs * spectrum) / max(total, 1e-6))
        return {"rms": rms, "zcr": zcr, "centroid": centroid}

    def _scores_from_features(self, feats: Mapping[str, float]) -> np.ndarray:
        rms = float(feats.get("rms", 0.0))
        zcr = float(feats.get("zcr", 0.0))
        centroid = float(feats.get("centroid", 0.0))

        # Normalise rough ranges so heuristics stay stable across devices.
        rms_n = min(1.0, rms / 0.3)  # speech typically <0.3 after scaling
        zcr_n = min(1.0, zcr / 0.5)
        centroid_n = min(1.0, centroid / 2500.0)  # 2.5kHz typical centroid upper bound

        scores = np.zeros(len(self.classes), dtype=np.float32)

        def idx(label: str) -> int:
            return self.classes.index(label)

        scores[idx("angry")] = 0.6 * rms_n + 0.4 * centroid_n
        scores[idx("surprise")] = 0.5 * centroid_n + 0.5 * zcr_n
        scores[idx("fear")] = 0.5 * centroid_n + 0.5 * (1.0 - zcr_n)
        scores[idx("sad")] = 0.6 * (1.0 - rms_n) + 0.4 * (1.0 - centroid_n)
        scores[idx("happy")] = 0.6 * zcr_n + 0.4 * rms_n
        scores[idx("disgust")] = 0.5 * rms_n + 0.5 * (1.0 - centroid_n)
        scores[idx("neutral")] = 0.3 * (1.0 - rms_n) + 0.3 * (1.0 - zcr_n) + 0.4 * (1.0 - centroid_n)

        # Ensure positivity then normalise.
        scores = np.maximum(scores, 1e-6)
        scores /= float(np.sum(scores))
        return scores

    def _smooth_probs(self, probs: np.ndarray) -> np.ndarray:
        self._prob_hist.append(probs)
        if len(self._prob_hist) == 1:
            return probs
        stacked = np.stack(list(self._prob_hist), axis=0)
        return np.mean(stacked, axis=0)

    # ---------------------------------------------------------------- inference
    def process(self, audio: Optional[np.ndarray], sr: Optional[int]) -> Dict[str, object]:
        """Process a new audio chunk (or return last state if audio missing)."""
        if audio is not None and audio.size:
            audio_f = self._ensure_float(np.asarray(audio))
            sr = int(sr or self.sample_rate)
            if sr != self.sample_rate:
                audio_f = self._linear_resample(audio_f, sr)
            self._update_buffer(audio_f)

            window = self._extract_window()
            feats = self._compute_features(window)
            raw_probs = self._scores_from_features(feats)
            probs = self._smooth_probs(raw_probs)

            conf = float(np.max(probs))
            self._state = SERState(probs=probs, conf=conf, features=feats, raw_probs=raw_probs)
        elif self._state is None:
            probs = np.zeros(len(self.classes), dtype=np.float32)
            probs[-1] = 1.0  # neutral
            feats = {"rms": 0.0, "zcr": 0.0, "centroid": 0.0}
            self._state = SERState(probs=probs, conf=1.0, features=feats, raw_probs=probs)

        if self._state is None:
            raise RuntimeError("SER state unavailable; call process with audio first")

        return self._state.to_dict()

    def reset(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)
        self._prob_hist.clear()
        self._state = None

    # Helpers for integration -------------------------------------------------
    def last_probs(self) -> Optional[np.ndarray]:
        return None if self._state is None else self._state.probs.copy()

    def last_conf(self) -> Optional[float]:
        return None if self._state is None else float(self._state.conf)

    def classes_list(self) -> List[str]:
        return list(self.classes)
