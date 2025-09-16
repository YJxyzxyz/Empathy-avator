# server/pipeline.py
# -*- coding: utf-8 -*-
from typing import Optional, Dict, Any, Tuple, List, Mapping
import time, uuid, wave, os, collections
import numpy as np

from models.fer.infer import FER
from models.ser import StreamingSER
from models.text import TextEmotion
from models.fusion import RuleFusion
from dialog.engine import DialogEngine
from dialog.policy import argmax_label, describe_style
from tts.synth import PiperTTS
from avatar.driver import viseme_from_audio

CLASSES = ("angry","disgust","fear","happy","sad","surprise","neutral")

DEFAULT_DIALOG_STYLE = describe_style("neutral")


def _write_wav_int16(path: str, pcm: np.ndarray, sr: int):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        if pcm.dtype != np.int16:
            pcm = pcm.astype(np.int16, copy=False)
        wf.writeframes(pcm.tobytes())


class Pipeline:
    """
    M3：FER + SER + 文本情感 + 规则式融合 + 检索式对话 + TTS/Avatar

    功能：
      - 视觉：FER（带多数投票平滑）
      - 音频：StreamingSER（RMS/ZCR/频谱质心启发式）
      - 文本：关键词情感识别，附带风险词提醒
      - 融合：按置信度门控的加权融合
      - 对话：检索增强对话引擎（关键词加权 RAG + 情绪风格）
      - 回复：情绪/文本联合触发 + 会话留存 + 数字人口型信息
    """

    def __init__(
        self,
        fer_onnx: str,
        piper_exe: str,
        piper_voice: str,
        audio_tmp_dir: str = "tmp_audio",
        tts_min_interval_sec: float = 2.0,
        smooth_window: int = 8,
        ser_sample_rate: int = 16000,
        ser_window: float = 1.2,
        fusion_weights: Optional[Mapping[str, float]] = None,
    ):
        self.fer = FER(fer_onnx, input_size=(224, 224), classes=CLASSES)
        self.ser = StreamingSER(sample_rate=ser_sample_rate, window_seconds=ser_window, classes=CLASSES)
        self.text_emotion = TextEmotion(classes=CLASSES)
        self.fusion = RuleFusion(classes=CLASSES, base_weights=fusion_weights)
        self.tts = PiperTTS(piper_exe=piper_exe, voice_path=piper_voice, out_dir=audio_tmp_dir)

        self.dialog = DialogEngine()
        self.history: List[Dict[str, Any]] = []
        self.max_history: int = 40
        self.turn_id_counter: int = 0
        self.current_turn_id: int = 0
        self.last_dialog_state: Dict[str, Any] = {
            "strategy": "idle",
            "triggers": ["boot"],
            "style": DEFAULT_DIALOG_STYLE,
            "clues": [],
            "sources": [],
            "risk": False,
            "source_id": None,
            "turn_id": 0,
            "history": [],
            "reply": "",
        }
        self.last_user_text: str = ""
        self.last_user_text_ts: float = 0.0

        # 情绪平滑
        self.smooth_window = int(smooth_window)
        self.emo_hist: "collections.deque[str]" = collections.deque(maxlen=self.smooth_window)
        self.stable_emo: Optional[str] = None   # 上一次“稳定情绪”

        # TTS 缓存
        self.last_reply: Optional[str] = None
        self.last_wav: Optional[str] = None
        self.last_sr: Optional[int] = None
        self.last_visemes: Optional[Any] = None
        self.last_tts_ts: float = 0.0
        self.tts_min_interval = float(tts_min_interval_sec)

    # ---------- 内部工具 ----------

    def _majority(self, labels: List[str]) -> Optional[str]:
        if not labels:
            return None
        vals, cnts = np.unique(np.array(labels), return_counts=True)
        return str(vals[np.argmax(cnts)])

    def _pcm_to_wavfile(self, pcm: np.ndarray, sr: int) -> str:
        token = uuid.uuid4().hex[:12] + ".wav"
        out_path = os.path.join(str(self.tts.out_dir), token)
        _write_wav_int16(out_path, pcm, sr)
        return token

    def _tts_once(self, reply: str) -> Tuple[str, int, Optional[Any]]:
        pcm, sr = self.tts.synth(reply)                 # 一次获取 PCM
        wav_file = self._pcm_to_wavfile(pcm, sr)        # 管线内写 wav
        try:
            vis = viseme_from_audio(pcm, sr)            # 嘴型可失败
        except Exception:
            vis = None
        self.last_reply, self.last_wav, self.last_sr, self.last_visemes = reply, wav_file, sr, vis
        self.last_tts_ts = time.time()
        return wav_file, sr, vis

    def _history_payload(self) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for turn in self.history[-self.max_history:]:
            if not isinstance(turn, dict):
                continue
            user = turn.get("user", {}) if isinstance(turn.get("user"), dict) else {}
            assistant = turn.get("assistant", {}) if isinstance(turn.get("assistant"), dict) else {}
            sources_list: List[Dict[str, Any]] = []
            for src in assistant.get("sources", []) or []:
                if isinstance(src, dict):
                    sources_list.append({k: v for k, v in src.items()})
            payload.append(
                {
                    "turn": int(turn.get("turn", 0)),
                    "ts": float(turn.get("ts", 0.0)),
                    "user": {
                        "text": str(user.get("text", "")),
                        "emo": user.get("emo"),
                        "ts": float(user.get("ts", 0.0)),
                    },
                    "assistant": {
                        "text": str(assistant.get("text", "")),
                        "emo": assistant.get("emo"),
                        "strategy": assistant.get("strategy"),
                        "triggers": list(assistant.get("triggers", [])),
                        "style": assistant.get("style"),
                        "clues": list(assistant.get("clues", [])),
                        "sources": sources_list,
                        "risk": bool(assistant.get("risk", False)),
                        "source_id": assistant.get("source_id"),
                        "ts": float(assistant.get("ts", 0.0)),
                    },
                }
            )
        return payload

    def _vision_branch(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        emo_out = self.fer.infer(frame_bgr)  # {'probs','conf','bbox'}
        probs = np.asarray(emo_out.get("probs", np.zeros(len(CLASSES), dtype=np.float32)))
        conf  = float(emo_out.get("conf", np.max(probs) if probs.size else 0.0))
        bbox  = emo_out.get("bbox", None)
        curr_emo = argmax_label(probs, CLASSES) if probs.size else "neutral"
        return {
            "probs": probs.tolist(),
            "conf": conf,
            "bbox": bbox,
            "emo": curr_emo,
        }

    def _audio_branch(self, audio_chunk: Optional[np.ndarray], audio_sr: Optional[int]) -> Dict[str, Any]:
        try:
            ser_out = self.ser.process(audio_chunk, audio_sr)
        except Exception:
            ser_out = {"probs": None, "conf": None, "features": None}
        else:
            probs = np.asarray(ser_out.get("probs", np.zeros(len(CLASSES), dtype=np.float32)))
            top = int(np.argmax(probs)) if probs.size else len(CLASSES) - 1
            ser_out = {
                **ser_out,
                "emo": CLASSES[top],
            }
        return ser_out

    def _text_branch(self, user_text: Optional[str]) -> Dict[str, Any]:
        text_out = self.text_emotion.analyse(user_text or "")
        probs = np.asarray(text_out.get("probs", np.zeros(len(CLASSES), dtype=np.float32)))
        top = int(np.argmax(probs)) if probs.size else len(CLASSES) - 1
        text_out = {
            **text_out,
            "emo": CLASSES[top],
        }
        return text_out

    # ---------- 主流程 ----------

    def step(
        self,
        frame_bgr: np.ndarray,
        audio_chunk: Optional[np.ndarray] = None,
        audio_sr: Optional[int] = None,
        user_text: Optional[str] = "",
        user_text_ts: Optional[float] = None,
    ) -> Dict[str, Any]:
        # 1) 各模态结果
        vision = self._vision_branch(frame_bgr)
        audio = self._audio_branch(audio_chunk, audio_sr)
        text = self._text_branch(user_text)
        risk_flag = bool(text.get("risk"))

        # 2) 情绪平滑基于融合后的标签
        modalities = {
            "vision": {"probs": vision.get("probs"), "conf": vision.get("conf")},
            "audio": audio,
            "text": text,
        }
        fusion = self.fusion.fuse(modalities)
        fused_label = fusion["emo"]
        fused_conf = fusion["conf"]
        self.emo_hist.append(fused_label)
        maj = self._majority(list(self.emo_hist)) or fused_label

        clean_text = (user_text or "").strip()
        text_ts = float(user_text_ts) if user_text_ts is not None else 0.0
        text_changed = False
        if clean_text:
            if text_ts and text_ts != self.last_user_text_ts:
                text_changed = True
            elif clean_text != self.last_user_text:
                text_changed = True

        triggers: List[str] = []
        if self.last_reply is None:
            triggers.append("warm_start")
        if risk_flag:
            triggers.append("safety_word")
        if text_changed:
            triggers.append("text_update")
        if self.stable_emo != maj:
            triggers.append("emotion_shift")

        need_new_reply = bool(triggers)

        if need_new_reply:
            dialog_turn = self.dialog.generate(
                user_text=clean_text,
                emo_label=maj,
                triggers=triggers,
                text_info=text,
                history=self.history,
                risk=risk_flag,
            )
            reply = dialog_turn.reply
            now = time.time()
            self.turn_id_counter += 1
            self.current_turn_id = self.turn_id_counter
            effective_ts = text_ts if text_ts > 0 else now
            turn_record = {
                "turn": self.current_turn_id,
                "ts": now,
                "user": {
                    "text": clean_text,
                    "emo": maj,
                    "ts": effective_ts,
                },
                "assistant": {
                    "text": reply,
                    "emo": maj,
                    "strategy": dialog_turn.strategy,
                    "triggers": list(dialog_turn.triggers),
                    "style": dialog_turn.style,
                    "clues": list(dialog_turn.clues),
                    "sources": [dict(src) for src in dialog_turn.sources],
                    "risk": dialog_turn.risk,
                    "source_id": dialog_turn.source_id,
                    "ts": now,
                },
            }
            self.history.append(turn_record)
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history:]
            self.stable_emo = maj
            self.last_user_text = clean_text
            self.last_user_text_ts = effective_ts
            dialog_payload = dialog_turn.to_payload()
            dialog_payload["turn_id"] = self.current_turn_id
            dialog_payload["history"] = self._history_payload()
            dialog_payload["reply"] = reply
            self.last_dialog_state = dialog_payload
        else:
            reply = self.last_reply if self.last_reply is not None else self.last_dialog_state.get("reply", "")
            dialog_payload = dict(self.last_dialog_state)
            dialog_payload.setdefault("turn_id", self.current_turn_id)
            dialog_payload.setdefault("strategy", "idle")
            dialog_payload.setdefault("triggers", [])
            dialog_payload.setdefault("style", DEFAULT_DIALOG_STYLE)
            dialog_payload.setdefault("clues", [])
            dialog_payload.setdefault("sources", [])
            dialog_payload.setdefault("risk", False)
            dialog_payload.setdefault("source_id", None)
            dialog_payload["history"] = self._history_payload()
            dialog_payload.setdefault("reply", reply)
            self.last_dialog_state["history"] = dialog_payload["history"]
            self.last_dialog_state["reply"] = dialog_payload["reply"]

        now = time.time()
        need_tts = (reply != self.last_reply)
        force_tts = need_new_reply and (reply != self.last_reply)

        if (need_tts and (force_tts or (now - self.last_tts_ts) >= self.tts_min_interval)) or self.last_wav is None or self.last_sr is None:
            wav_file, sr, vis = self._tts_once(reply)
        else:
            wav_file, sr, vis = self.last_wav, self.last_sr, self.last_visemes

        dialog_payload["turn_id"] = dialog_payload.get("turn_id", self.current_turn_id)
        dialog_payload["reply"] = reply

        return {
            "emo": maj,           # 平滑后的稳定情绪
            "conf": fused_conf,
            "reply": reply,
            "wav": wav_file,
            "sr": sr,
            "bbox": vision.get("bbox"),
            "visemes": vis,
            "turn_id": dialog_payload["turn_id"],
            "modalities": {
                "vision": vision,
                "audio": audio,
                "text": {**text, "text": user_text or "", "ts": text_ts},
            },
            "fusion": fusion,
            "dialog": dialog_payload,
        }
