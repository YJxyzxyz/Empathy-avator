# server/pipeline.py
# -*- coding: utf-8 -*-
from typing import Optional, Dict, Any, Tuple, List, Mapping
import time, uuid, wave, os, collections
import numpy as np

from models.fer.infer import FER
from models.ser import StreamingSER
from models.text import TextEmotion
from models.fusion import RuleFusion
from dialog.policy import argmax_label
from dialog import DialogEngine
from tts.synth import PiperTTS
from avatar.driver import viseme_from_audio

CLASSES = ("angry","disgust","fear","happy","sad","surprise","neutral")


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
    M3：三模态识别 + 对话策略 + 数字人口型同步

    功能：
      - 视觉：FER（带多数投票平滑）
      - 音频：StreamingSER（RMS/ZCR/频谱质心启发式）
      - 文本：关键词情感识别，附带风险词提醒
      - 融合：按置信度门控的加权融合
      - 对话：检索式策略 + 风险兜底
      - 数字人：TTS + viseme 序列输出，供前端驱动口型
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
        self.last_spoken_reply: Optional[str] = None
        self.last_turn_id: Optional[str] = None
        self.last_dialog_meta: Optional[Dict[str, Any]] = None
        self.last_reply_ts: float = 0.0

        # 文本缓存
        self._last_user_text: Optional[str] = None

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

    def _tts_once(self, reply: str) -> Tuple[str, int, Optional[Any], str]:
        pcm, sr = self.tts.synth(reply)                 # 一次获取 PCM
        wav_file = self._pcm_to_wavfile(pcm, sr)        # 管线内写 wav
        try:
            vis = viseme_from_audio(pcm, sr)            # 嘴型可失败
        except Exception:
            vis = None
        turn_id = wav_file.split(".")[0]
        self.last_spoken_reply = reply
        self.last_reply = reply
        self.last_wav, self.last_sr, self.last_visemes = wav_file, sr, vis
        self.last_tts_ts = time.time()
        self.last_turn_id = turn_id
        self.last_reply_ts = self.last_tts_ts
        return wav_file, sr, vis, turn_id

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
    ) -> Dict[str, Any]:
        # 1) 各模态结果
        vision = self._vision_branch(frame_bgr)
        audio = self._audio_branch(audio_chunk, audio_sr)
        text = self._text_branch(user_text)

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

        # 3) 仅当“稳定情绪”发生改变或文本变化时，才更新 reply
        need_new_reply = (self.stable_emo != maj)
        text_changed = (user_text or "") != (self._last_user_text or "")
        if need_new_reply:
            self.stable_emo = maj
        if text_changed:
            self._last_user_text = user_text or ""

        trigger_reasons: List[str] = []
        if need_new_reply:
            trigger_reasons.append("emotion_shift")
        if text_changed:
            trigger_reasons.append("user_text")

        if trigger_reasons or self.last_reply is None:
            reply, dialog_meta = self.dialog.generate(
                user_text or "",
                maj,
                fused_conf=fused_conf,
                modalities=modalities,
            )
            if trigger_reasons:
                dialog_meta["trigger"] = trigger_reasons
            self.last_dialog_meta = dialog_meta
            self.last_reply = reply
        else:
            reply = self.last_reply
            dialog_meta = self.last_dialog_meta or {}

        # 4) TTS 节流：reply 变化 且 距上次合成 >= 最小间隔（文本变化可立即触发）
        now = time.time()
        need_tts = (
            reply != self.last_spoken_reply
            and (
                (now - self.last_tts_ts) >= self.tts_min_interval
                or text_changed
            )
        )

        if need_tts or self.last_wav is None or self.last_sr is None:
            wav_file, sr, vis, turn_id = self._tts_once(reply)
        else:
            wav_file, sr, vis, turn_id = self.last_wav, self.last_sr, self.last_visemes, self.last_turn_id

        # 5) 返回
        return {
            "emo": maj,           # 平滑后的稳定情绪
            "conf": fused_conf,
            "reply": reply,
            "wav": wav_file,
            "sr": sr,
            "bbox": vision.get("bbox"),
            "visemes": vis,
            "turn_id": turn_id,
            "reply_ts": self.last_reply_ts,
            "modalities": {
                "vision": vision,
                "audio": audio,
                "text": {**text, "text": user_text or ""},
            },
            "fusion": fusion,
            "dialog": dialog_meta,
        }
