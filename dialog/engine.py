"""Simple retrieval-based dialog engine for M3 milestone."""

from __future__ import annotations

import collections
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from .policy import EMO_STYLES, make_reply, is_risky


_WORD_RE = re.compile(r"[\u4e00-\u9fa5]+|[A-Za-z0-9']+")


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [tok.lower() for tok in _WORD_RE.findall(str(text)) if tok.strip()]


def _counter(tokens: Sequence[str]) -> Dict[str, float]:
    freq: Dict[str, float] = {}
    for tok in tokens:
        freq[tok] = freq.get(tok, 0.0) + 1.0
    return freq


def _norm(counter: Dict[str, float]) -> float:
    return math.sqrt(sum(v * v for v in counter.values())) or 1.0


@dataclass
class KnowledgeEntry:
    """A lightweight entry in the retrieval table."""

    id: str
    prompt: str
    response: str
    emotion: str
    follow_up: str = ""
    tags: Tuple[str, ...] = ()
    _tokens: Tuple[str, ...] = ()
    _counter: Optional[Dict[str, float]] = None
    _norm: float = 1.0

    def with_cache(self) -> "KnowledgeEntry":
        tokens = tuple(_tokenize(self.prompt) + list(self.tags))
        self._tokens = tokens
        self._counter = _counter(tokens)
        self._norm = _norm(self._counter)
        return self


DEFAULT_KNOWLEDGE: Tuple[KnowledgeEntry, ...] = (
    KnowledgeEntry(
        id="work_pressure",
        prompt="最近工作压力太大，总是被批评，还得不断加班",
        response="听起来工作上的压力让你身心都很疲惫。或许我们可以先把让你最难受的部分说清楚，再一起想想可以调整的节奏。",
        follow_up="有没有一些小的休息方式是你觉得有效的？",
        emotion="sad",
        tags=("压力", "加班", "批评"),
    ),
    KnowledgeEntry(
        id="study_anxiety",
        prompt="考试复习时总是紧张，什么都记不住",
        response="紧张会让大脑更难专注，这是很常见的反应。也许我们可以尝试把复习拆成更小的步骤，一次只处理一个部分。",
        follow_up="你觉得从哪一科或者哪个章节开始会最容易一些？",
        emotion="fear",
        tags=("考试", "学习", "焦虑"),
    ),
    KnowledgeEntry(
        id="relationship_conflict",
        prompt="和伴侣吵架后觉得很受伤，也不知道怎么修复",
        response="关系里的冲突常常让人心里很难受，你愿意讲讲争执里让你最在意的点吗？这样我们可以看看是否有表达的空间。",
        follow_up="在这段关系里，什么事情会让你感到被理解？",
        emotion="sad",
        tags=("关系", "伴侣", "争吵"),
    ),
    KnowledgeEntry(
        id="lonely",
        prompt="最近总感觉没人可以倾诉，很孤独",
        response="孤独感真的会让人觉得无助。谢谢你愿意在这里说出来，我们可以一起探讨什么样的连接或活动能给你带来一些陪伴感。",
        follow_up="过去有没有让你觉得温暖或被支持的人或事情？",
        emotion="sad",
        tags=("孤独", "倾诉", "朋友"),
    ),
    KnowledgeEntry(
        id="sleep_issue",
        prompt="经常睡不着，脑子总是停不下来",
        response="睡眠困难在压力大的时候很常见。或许我们可以先理清晚上反复出现的念头，再尝试一些放松的小练习。",
        follow_up="有没有一些在你身上起作用的放松方式？哪怕是很小的也可以。",
        emotion="fear",
        tags=("失眠", "睡眠", "压力"),
    ),
    KnowledgeEntry(
        id="anger_unfair",
        prompt="觉得被领导不公平对待，很生气",
        response="被不公平对待会让人感到愤怒是可以理解的。我们可以先梳理一下发生了什么，再看看哪些部分是你希望表达或者改变的。",
        follow_up="如果可以向对方表达，你最想让对方理解的是什么？",
        emotion="angry",
        tags=("愤怒", "不公平", "领导"),
    ),
    KnowledgeEntry(
        id="surprise_change",
        prompt="突然被通知要调岗，一时间接受不了",
        response="突如其来的变化确实会让人措手不及。我们可以先搞清楚这次变化对你意味着什么，再慢慢理顺接下来的安排。",
        follow_up="在新的变化里，你最担心或者最期待的部分分别是什么？",
        emotion="surprise",
        tags=("变化", "调岗", "突然"),
    ),
    KnowledgeEntry(
        id="achieve_share",
        prompt="今天终于完成了一个长期目标，很开心",
        response="恭喜你达成这个目标！在忙碌之后体验到成就感真的很棒。愿意多讲讲这个旅程中哪些瞬间让你最感动吗？",
        follow_up="你希望如何好好庆祝一下自己呢？",
        emotion="happy",
        tags=("开心", "成功", "目标"),
    ),
    KnowledgeEntry(
        id="disgust_environment",
        prompt="身边的环境让我很厌烦，总是想逃离",
        response="当环境让人感到厌烦时，保持自我界限是很重要的。我们可以一起看看有哪些部分是可以调整的，哪些需要先保护好自己。",
        follow_up="有没有一些小的改变能让你暂时缓解这种不适感？",
        emotion="disgust",
        tags=("环境", "厌烦", "界限"),
    ),
    KnowledgeEntry(
        id="self_doubt",
        prompt="总觉得自己不够好，对未来没信心",
        response="质疑自己是很多人在面对压力时都会经历的。也许我们可以一起回顾一些你曾经的力量或成功经验，帮助你重新看见自己。",
        follow_up="回想一下，最近有没有哪件小事体现了你的能力或坚持？",
        emotion="neutral",
        tags=("自我怀疑", "未来", "信心"),
    ),
    KnowledgeEntry(
        id="fear_uncertain",
        prompt="面对未知的事情很害怕，不知道该怎么办",
        response="对未知感到害怕是人之常情，我们可以先把你所知道的部分列出来，看看哪些是可以慢慢掌控的。",
        follow_up="如果把事情拆成几个小步骤，你想先从哪一步开始？",
        emotion="fear",
        tags=("害怕", "未知", "怎么办"),
    ),
)  # type: ignore


DEFAULT_KNOWLEDGE = tuple(entry.with_cache() for entry in DEFAULT_KNOWLEDGE)


class ResponseRetriever:
    """A tiny TF-like retriever built on bag-of-words cosine similarity."""

    def __init__(self, entries: Optional[Iterable[KnowledgeEntry]] = None):
        prepared: List[KnowledgeEntry] = []
        for entry in (entries or DEFAULT_KNOWLEDGE):
            if entry._counter is None:
                prepared.append(entry.with_cache())
            else:
                prepared.append(entry)
        self.entries = tuple(prepared)

    def match(
        self,
        query: str,
        emotion: str,
        history_texts: Sequence[str] = (),
    ) -> Optional[Dict[str, Any]]:
        tokens = _tokenize(query)
        if not tokens:
            return None

        q_counter = _counter(tokens)
        q_norm = _norm(q_counter)

        # Context bonus: overlap with last user turns
        context_tokens: List[str] = []
        for text in history_texts[-3:]:
            context_tokens.extend(_tokenize(text))
        context_set = set(context_tokens)

        best: Optional[Dict[str, Any]] = None
        best_score = 0.0

        for entry in self.entries:
            shared = set(entry._tokens) & set(tokens)
            if not shared:
                continue
            dot = sum(q_counter.get(tok, 0.0) * entry._counter.get(tok, 0.0) for tok in shared)
            score = dot / (q_norm * entry._norm + 1e-6)
            if entry.emotion == emotion:
                score *= 1.12
            if context_set:
                overlap = len(context_set & set(entry._tokens))
                score *= 1.0 + overlap * 0.04

            if score > best_score:
                best_score = score
                best = {
                    "entry": entry,
                    "score": float(score),
                    "shared_tokens": sorted(shared),
                    "query_tokens": tokens,
                }

        return best


class DialogEngine:
    """Manage conversation history and generate responses."""

    def __init__(
        self,
        knowledge: Optional[Iterable[KnowledgeEntry]] = None,
        history_turns: int = 10,
    ):
        self.retriever = ResponseRetriever(entries=knowledge)
        self.history: Deque[Dict[str, Any]] = collections.deque(maxlen=history_turns)
        self.turn_index = 0

    # ------------------------------------------------------------------
    def reset(self):
        self.history.clear()
        self.turn_index = 0

    # ------------------------------------------------------------------
    def _append_history(self, role: str, text: str, emotion: str):
        if not text:
            return
        self.history.append(
            {
                "ts": time.time(),
                "role": role,
                "text": text,
                "emotion": emotion,
            }
        )

    def _user_history_texts(self) -> List[str]:
        return [turn["text"] for turn in self.history if turn.get("role") == "user"]

    def _style_wrap(self, text: str, emo_label: str) -> str:
        style = EMO_STYLES.get(emo_label, "平静、中性")
        if text.startswith("["):
            return text
        return f"[{style}] {text}".strip()

    # ------------------------------------------------------------------
    def generate(
        self,
        user_text: str,
        emo_label: str,
        fused_conf: Optional[float] = None,
        modalities: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        modalities = modalities or {}
        clean_text = (user_text or "").strip()
        diag: Dict[str, Any] = {
            "emotion": emo_label,
            "input": clean_text,
            "fused_conf": fused_conf,
            "modalities": list(modalities.keys()),
        }

        risk = is_risky(clean_text)
        diag["risk"] = risk

        if risk:
            reply = make_reply(clean_text, emo_label)
            diag.update({
                "source": "safety",
                "note": "触发风险词应急回复",
            })
        else:
            retrieval = None
            if clean_text:
                retrieval = self.retriever.match(
                    clean_text,
                    emo_label,
                    history_texts=self._user_history_texts(),
                )

            if retrieval:
                entry: KnowledgeEntry = retrieval["entry"]
                message = entry.response
                if fused_conf is not None and fused_conf < 0.35:
                    message += "。我也想确认下我的理解，如果有不准确的地方请你提醒我。"
                if entry.follow_up:
                    message += " " + entry.follow_up
                reply = self._style_wrap(message, emo_label)
                diag.update(
                    {
                        "source": "retrieval",
                        "match_id": entry.id,
                        "match_prompt": entry.prompt,
                        "score": retrieval.get("score"),
                        "shared_tokens": retrieval.get("shared_tokens"),
                        "follow_up": entry.follow_up,
                    }
                )
            else:
                reply = make_reply(clean_text, emo_label)
                diag.update({
                    "source": "rule",
                    "note": "使用规则库生成",
                })

        self._append_history("user", clean_text, emo_label)
        self._append_history("assistant", reply, emo_label)
        self.turn_index += 1
        diag["turn_index"] = self.turn_index
        diag["history_size"] = len(self.history)
        diag["style"] = EMO_STYLES.get(emo_label, "平静、中性")
        diag["timestamp"] = time.time()
        return reply, diag

