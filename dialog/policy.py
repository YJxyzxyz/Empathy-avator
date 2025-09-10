# -*- coding: utf-8 -*-
from typing import Optional

EMO_STYLES = {
    "happy": "轻快、鼓励",
    "neutral": "平静、中性",
    "sad": "温柔、共情",
    "angry": "冷静、缓和",
    "fear": "安抚、安全感",
    "surprise": "克制、解释",
    "disgust": "尊重、转焦点",
    "confused": "澄清、引导",
}

# 风险/求助词（可扩充）
RISK_WORDS = {"自杀","轻生","伤害自己","不想活","绝望","自残","无意义","结束生命","跳楼"}

# 基础模板（按需扩充）
BASE_REPLY = {
    "happy":    "听起来你有一些积极的感受，愿意分享让你开心的事吗？",
    "sad":      "我能理解你的难过。我在这里，愿意听你慢慢说。",
    "angry":    "你的感受很重要。我们可以一起梳理下让你生气的点吗？",
    "fear":     "你是安全的，我们一步一步来。要不要从你最担心的一点开始？",
    "confused": "我可能需要再确认一下：你最在意的是哪一部分？可以多说一点吗？",
    "surprise": "我明白，这有些出乎意料。要不要一起回顾发生了什么？",
    "disgust":  "我理解你的不适。也许我们可以将注意力放在你能掌控的部分上。",
    "neutral":  "我在认真听，你可以随意展开说说。",
}

def argmax_label(probs, classes):
    import numpy as np
    return classes[int(np.argmax(probs))]

def is_risky(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.strip()
    if not t:
        return False
    return any(k in t for k in RISK_WORDS)

def make_reply(user_text: Optional[str], emo_label: str) -> str:
    """基于情绪标签与规则生成可用回复"""
    if is_risky(user_text):
        return ("我在，先保证你的安全。如果你有伤害自己的想法，请立即联系当地的紧急援助或可信任的人。"
                "如果愿意，我们可以一起寻找能够帮助你的资源。你不是一个人。")

    style = EMO_STYLES.get(emo_label, "平静、中性")
    base = BASE_REPLY.get(emo_label, BASE_REPLY["neutral"])
    return f"[{style}] {base}"
