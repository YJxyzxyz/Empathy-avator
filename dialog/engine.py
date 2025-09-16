# -*- coding: utf-8 -*-
"""Lightweight retrieval-augmented dialog engine for Milestone 3."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from dialog.policy import BASE_REPLY, describe_style, emergency_reply


@dataclass
class KnowledgeEntry:
    """In-memory knowledge snippet used by the retrieval heuristic."""

    id: str
    title: str
    response: str
    keywords: Mapping[str, float]
    emotions: Sequence[str] = field(default_factory=tuple)
    followup: Optional[str] = None
    category: Optional[str] = None


@dataclass
class DialogTurn:
    """Structured result returned by :class:`DialogEngine`."""

    reply: str
    strategy: str
    triggers: List[str]
    style: str
    clues: List[str] = field(default_factory=list)
    sources: List[Dict[str, object]] = field(default_factory=list)
    risk: bool = False
    source_id: Optional[str] = None

    def to_payload(self) -> Dict[str, object]:
        return {
            "strategy": self.strategy,
            "triggers": list(self.triggers),
            "style": self.style,
            "clues": list(self.clues),
            "sources": [dict(src) for src in self.sources],
            "risk": bool(self.risk),
            "source_id": self.source_id,
        }


class DialogEngine:
    """Keyword-weighted retrieval engine for local empathetic replies."""

    def __init__(self, *, top_k: int = 3) -> None:
        self.top_k = int(top_k)
        self.knowledge: List[KnowledgeEntry] = self._build_knowledge()

    # ------------------------------------------------------------------ helpers
    def _build_knowledge(self) -> List[KnowledgeEntry]:
        """Static lightweight knowledge base used for retrieval."""
        return [
            KnowledgeEntry(
                id="stress_breath",
                title="呼吸调节与减压",
                response="听上去压力有些沉重，可以尝试慢慢吸气四拍、停两拍、再缓缓呼气六拍，让身体找到节奏。",
                followup="如果愿意，也可以告诉我最近最让你紧张的一件小事，我们一起想想如何放松。",
                keywords={"压力": 1.2, "焦虑": 1.2, "紧张": 1.0, "慌": 0.8, "崩溃": 0.9},
                emotions=("fear", "sad", "angry"),
                category="情绪调节",
            ),
            KnowledgeEntry(
                id="lonely_support",
                title="孤独感与社会支持",
                response="你提到的那份孤单让我很心疼。可以尝试联系一位信任的人，哪怕只是发一条信息，也能提醒自己并不孤立。",
                followup="如果你愿意，我们也可以一起规划一个小小的行动，比如给谁发消息或做一件让自己安心的事情。",
                keywords={"孤单": 1.3, "孤独": 1.3, "没人": 1.1, "朋友": 0.9, "陪伴": 1.0, "一个人": 1.0},
                emotions=("sad", "neutral"),
                category="关系支持",
            ),
            KnowledgeEntry(
                id="anger_channel",
                title="情绪宣泄与界限",
                response="被这样的事情惹恼是可以理解的。可以先在安全的空间写下让你生气的点，再想想哪些是你能掌控、可以调整的。",
                followup="需要时也可以练习几次深呼吸或短暂离开现场，给自己一点缓冲时间。",
                keywords={"生气": 1.3, "愤怒": 1.3, "烦": 1.0, "委屈": 1.0, "不公平": 1.1, "气愤": 1.2},
                emotions=("angry", "disgust"),
                category="情绪调节",
            ),
            KnowledgeEntry(
                id="confused_direction",
                title="迷茫与决策",
                response="面对未知感到迷茫很正常。可以把现在最困扰你的三件事情写下来，按照“紧急度”和“可控度”分别排序。",
                followup="我们也可以从其中最容易尝试的小步骤开始，逐步找回掌控感。",
                keywords={"迷茫": 1.4, "困惑": 1.2, "不知道": 1.0, "怎么办": 1.0, "方向": 1.0, "犹豫": 0.9},
                emotions=("confused", "neutral", "sad"),
                category="自我探索",
            ),
            KnowledgeEntry(
                id="fear_safety",
                title="安全感与焦虑缓解",
                response="害怕的感觉说明你在认真保护自己。可以观察一下身体哪些部位绷得最紧，尝试从脚到头一点点放松。",
                followup="如果某些情境让你特别不安，也欢迎告诉我，我们可以一起制定应对小方案。",
                keywords={"害怕": 1.4, "担心": 1.1, "恐惧": 1.4, "不安": 1.1, "紧张": 0.9},
                emotions=("fear", "sad"),
                category="安全感",
            ),
            KnowledgeEntry(
                id="happy_share",
                title="积极体验与分享",
                response="听到你有好消息我也为你开心！把这份喜悦分享给重要的人，会让幸福持续更久。",
                followup="也可以记录下让你感到满足的细节，日后回顾时能重新点亮心情。",
                keywords={"开心": 1.2, "高兴": 1.2, "激动": 1.1, "兴奋": 1.0, "收获": 0.9, "成功": 1.0},
                emotions=("happy", "surprise"),
                category="积极体验",
            ),
        ]

    def _context_text(self, history: Optional[Sequence[Mapping[str, object]]], *, max_turns: int = 3) -> str:
        if not history:
            return ""
        texts: List[str] = []
        for turn in history[-max_turns:]:
            user = turn.get("user") if isinstance(turn, Mapping) else None
            if isinstance(user, Mapping):
                txt = str(user.get("text", "")).strip()
                if txt:
                    texts.append(txt)
        return "。".join(texts)

    def _used_entry_ids(self, history: Optional[Sequence[Mapping[str, object]]]) -> List[str]:
        ids: List[str] = []
        if not history:
            return ids
        for turn in history:
            assistant = turn.get("assistant") if isinstance(turn, Mapping) else None
            if isinstance(assistant, Mapping):
                sid = assistant.get("source_id")
                if isinstance(sid, str):
                    ids.append(sid)
        return ids

    def _score_entry(
        self,
        entry: KnowledgeEntry,
        *,
        text: str,
        context: str,
        emo: str,
        used_ids: Iterable[str],
    ) -> Tuple[float, List[str]]:
        if not text and not context:
            return 0.0, []
        score = 0.0
        matched: List[str] = []
        used = set(used_ids)
        for kw, weight in entry.keywords.items():
            hit = text.count(kw)
            if context:
                hit += int(round(context.count(kw) * 0.4))
            if hit > 0:
                matched.append(kw)
                score += float(weight) * hit
        if not matched:
            return 0.0, []
        if entry.emotions and emo in entry.emotions:
            score *= 1.2
        if entry.id in used:
            score *= 0.6
        return score, matched

    def _extract_text_clues(self, text_info: Optional[Mapping[str, object]]) -> List[str]:
        if not text_info:
            return []
        matched = text_info.get("matched")
        if not isinstance(matched, Mapping):
            return []
        clues = [k for k, v in matched.items() if isinstance(v, (int, float)) and v > 0]
        return clues[:5]

    def _bridge_phrase(self, triggers: Sequence[str], history: Optional[Sequence[Mapping[str, object]]]) -> str:
        parts: List[str] = []
        trig = set(triggers)
        if "warm_start" in trig:
            parts.append("很高兴在这里与你连线。")
        if "text_update" in trig:
            parts.append("谢谢你补充这些感受，我会认真听。")
        if "emotion_shift" in trig and history:
            parts.append("我也注意到情绪有了一点变化，我们可以慢慢梳理。")
        return " ".join(parts)

    # ------------------------------------------------------------------- public
    def generate(
        self,
        *,
        user_text: str,
        emo_label: str,
        triggers: Sequence[str],
        text_info: Optional[Mapping[str, object]] = None,
        history: Optional[Sequence[Mapping[str, object]]] = None,
        risk: bool = False,
    ) -> DialogTurn:
        clean_text = (user_text or "").strip()
        style = describe_style(emo_label)

        if risk:
            reply = emergency_reply()
            clues = self._extract_text_clues(text_info)
            return DialogTurn(
                reply=reply,
                strategy="emergency",
                triggers=list(triggers),
                style=style,
                clues=clues,
                sources=[],
                risk=True,
                source_id="emergency",
            )

        base_prompt = BASE_REPLY.get(emo_label, BASE_REPLY["neutral"])
        bridge = self._bridge_phrase(triggers, history)
        if bridge:
            preface = f"[{style}] {bridge} {base_prompt}"
        else:
            preface = f"[{style}] {base_prompt}"

        context = self._context_text(history)
        used_ids = self._used_entry_ids(history)

        scored: List[Tuple[float, List[str], KnowledgeEntry]] = []
        for entry in self.knowledge:
            score, matched = self._score_entry(
                entry,
                text=clean_text,
                context=context,
                emo=emo_label,
                used_ids=used_ids,
            )
            if score <= 0:
                continue
            scored.append((score, matched, entry))

        scored.sort(key=lambda item: item[0], reverse=True)

        if scored:
            _, top_keywords, top_entry = scored[0]
            sources: List[Dict[str, object]] = []
            for score, matched, entry in scored[: self.top_k]:
                sources.append(
                    {
                        "id": entry.id,
                        "title": entry.title,
                        "score": round(float(score), 3),
                        "keywords": list(matched),
                        "summary": entry.response,
                        "category": entry.category,
                    }
                )
            body = top_entry.response
            follow = top_entry.followup or ""
            reply = f"{preface} {body}"
            if follow:
                reply = f"{reply} {follow}"
            clues = list(dict.fromkeys(top_keywords + self._extract_text_clues(text_info)))
            return DialogTurn(
                reply=reply,
                strategy="retrieval",
                triggers=list(triggers),
                style=style,
                clues=clues,
                sources=sources,
                risk=False,
                source_id=top_entry.id,
            )

        fallback = preface
        clues = self._extract_text_clues(text_info)
        return DialogTurn(
            reply=fallback,
            strategy="fallback",
            triggers=list(triggers),
            style=style,
            clues=clues,
            sources=[],
            risk=False,
            source_id=None,
        )
