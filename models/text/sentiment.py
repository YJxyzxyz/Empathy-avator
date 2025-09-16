# -*- coding: utf-8 -*-
"""Rule-based text emotion estimation.

This module provides a light-weight alternative to a full NLP sentiment model.
It scans the user utterance for curated keywords and maps them to the same
emotion set used across the project.  Although simple, it enables the
multimodal fusion pipeline to react to textual cues and also acts as a drop-in
placeholder for future learned models.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

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
class TextEmotionResult:
    probs: np.ndarray
    conf: float
    matched: Dict[str, int]
    risk: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "probs": self.probs.tolist(),
            "conf": float(self.conf),
            "matched": {k: int(v) for k, v in self.matched.items()},
            "risk": bool(self.risk),
        }


class TextEmotion:
    """Simple lexicon-based text emotion recogniser."""

    def __init__(
        self,
        classes: Sequence[str] = DEFAULT_CLASSES,
        neutral_boost: float = 0.2,
    ) -> None:
        self.classes: Sequence[str] = tuple(classes)
        self.lexicon: Dict[str, Sequence[str]] = self._build_lexicon()
        self.risk_words = {
            "自杀",
            "轻生",
            "结束生命",
            "不想活",
            "跳楼",
            "绝望",
            "自残",
            "伤害自己",
        }
        self.neutral_boost = float(neutral_boost)
        self._pattern_cache: Dict[str, re.Pattern[str]] = {}

    # ---------------------------------------------------------------- private
    def _build_lexicon(self) -> Dict[str, Sequence[str]]:
        return {
            "angry": ("生气", "愤怒", "烦", "火大", "受不了", "气愤", "恼火", "爆炸"),
            "disgust": ("恶心", "讨厌", "厌恶", "反感", "脏", "嫌弃"),
            "fear": ("害怕", "恐惧", "担心", "焦虑", "紧张", "慌", "吓"),
            "happy": ("开心", "高兴", "激动", "快乐", "兴奋", "满意", "轻松"),
            "sad": ("难过", "伤心", "想哭", "失落", "沮丧", "委屈", "痛苦"),
            "surprise": ("惊讶", "意外", "没想到", "震惊", "不可思议"),
            "neutral": ("还好", "一般", "正常", "无所谓", "没事"),
        }

    def _get_pattern(self, word: str) -> re.Pattern[str]:
        if word not in self._pattern_cache:
            self._pattern_cache[word] = re.compile(re.escape(word))
        return self._pattern_cache[word]

    def _match_counts(self, text: str) -> Dict[str, int]:
        counts: Dict[str, int] = {label: 0 for label in self.classes}
        for label, words in self.lexicon.items():
            for word in words:
                pattern = self._get_pattern(word)
                matches = pattern.findall(text)
                if matches:
                    counts[label] += len(matches)
        return counts

    def _scores_from_counts(self, counts: Mapping[str, int]) -> np.ndarray:
        scores = np.ones(len(self.classes), dtype=np.float32) * 1e-6
        total_hits = 0
        for idx, label in enumerate(self.classes):
            hit = int(counts.get(label, 0))
            scores[idx] += float(hit)
            total_hits += hit
        if total_hits == 0:
            neutral_idx = self.classes.index("neutral")
            scores[neutral_idx] += self.neutral_boost
        scores /= float(np.sum(scores))
        return scores

    # ----------------------------------------------------------------- public
    def analyse(self, text: Optional[str]) -> Dict[str, object]:
        if text is None or not text.strip():
            probs = np.zeros(len(self.classes), dtype=np.float32)
            probs[self.classes.index("neutral")] = 1.0
            result = TextEmotionResult(
                probs=probs,
                conf=1.0,
                matched={label: 0 for label in self.classes},
                risk=False,
            )
            return result.to_dict()

        counts = self._match_counts(text)
        probs = self._scores_from_counts(counts)
        conf = float(np.max(probs))

        risk = any(word in text for word in self.risk_words)

        result = TextEmotionResult(
            probs=probs,
            conf=conf,
            matched=counts,
            risk=risk,
        )
        return result.to_dict()
