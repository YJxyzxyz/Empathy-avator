# server/pipeline.py
# -*- coding: utf-8 -*-
from typing import Optional, Dict, Any, Tuple, List
import time, uuid, wave, os, collections
import numpy as np

from models.fer.infer import FER
from dialog.policy import make_reply, argmax_label
from tts.synth import PiperTTS
from avatar.driver import viseme_from_audio

CLASSES = ('angry','disgust','fear','happy','sad','surprise','neutral')

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
    M1：FER -> 策略 -> 单次TTS（PCM）-> 本地写wav
    加入：
      - 情绪平滑（多数投票）
      - 仅在“稳定情绪改变”时更新 reply
      - TTS 双阈值：reply 变化 且 超过最小间隔 才合成
    """
    def __init__(self,
                 fer_onnx: str,
                 piper_exe: str,
                 piper_voice: str,
                 audio_tmp_dir: str = "tmp_audio",
                 tts_min_interval_sec: float = 2.0,
                 smooth_window: int = 8):
        self.fer = FER(fer_onnx, input_size=(224, 224), classes=CLASSES)
        self.tts = PiperTTS(piper_exe=piper_exe, voice_path=piper_voice, out_dir=audio_tmp_dir)

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

    # ---------- 主流程 ----------

    def step(self, frame_bgr: np.ndarray, user_text: Optional[str] = "") -> Dict[str, Any]:
        # 1) FER
        emo_out = self.fer.infer(frame_bgr)  # {'probs','conf','bbox'}
        probs = np.asarray(emo_out.get("probs", np.zeros(len(CLASSES), dtype=np.float32)))
        conf  = float(emo_out.get("conf", np.max(probs) if probs.size else 0.0))
        bbox  = emo_out.get("bbox", None)
        curr_emo = argmax_label(probs, CLASSES) if probs.size else "neutral"

        # 2) 情绪平滑（更新历史，求多数）
        self.emo_hist.append(curr_emo)
        maj = self._majority(list(self.emo_hist))
        if maj is None:
            maj = curr_emo

        # 3) 仅当“稳定情绪”发生改变时，才更新 reply
        need_new_reply = (self.stable_emo != maj)
        if need_new_reply:
            self.stable_emo = maj

        # 生成（或复用）reply
        reply = self.last_reply if not need_new_reply and self.last_reply else make_reply(user_text or "", maj)

        # 4) TTS 节流：reply 变化 且 距上次合成 >= 最小间隔
        now = time.time()
        need_tts = (reply != self.last_reply) and ((now - self.last_tts_ts) >= self.tts_min_interval)

        if need_tts or self.last_wav is None or self.last_sr is None:
            wav_file, sr, vis = self._tts_once(reply)
        else:
            wav_file, sr, vis = self.last_wav, self.last_sr, self.last_visemes

        # 5) 返回
        return {
            "emo": maj,           # 平滑后的稳定情绪
            "conf": conf,
            "reply": reply,
            "wav": wav_file,
            "sr": sr,
            "bbox": bbox,
            "visemes": vis,
        }
