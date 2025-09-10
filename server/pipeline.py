# server/pipeline.py
# -*- coding: utf-8 -*-
from typing import Optional, Dict, Any, Tuple
import time
import numpy as np

from models.fer.infer import FER
from dialog.policy import make_reply, argmax_label
from tts.synth import PiperTTS
from avatar.driver import viseme_from_audio

# 与训练一致：FER7
CLASSES = ('angry','disgust','fear','happy','sad','surprise','neutral')


class Pipeline:
    """
    M1 推理管线（带 TTS 缓存 & 频率限制）：
      1) FER：单帧表情识别
      2) Policy：基于表情/可选文本生成一句回复
      3) TTS：仅在回复变化或到达最小间隔时才重新合成；否则复用上次 wav
      4) 可选：根据 PCM 生成嘴型曲线（缓存）
    返回字段与前端/后端 API 对齐：emo / conf / reply / wav / sr / bbox / visemes
    """

    def __init__(self,
                 fer_onnx: str,
                 piper_exe: str,
                 piper_voice: str,
                 audio_tmp_dir: str = "tmp_audio",
                 tts_min_interval_sec: float = 2.0):
        # 模块初始化
        self.fer = FER(fer_onnx, input_size=(224, 224), classes=CLASSES)
        self.tts = PiperTTS(piper_exe=piper_exe, voice_path=piper_voice, out_dir=audio_tmp_dir)

        # TTS 缓存（上一次的回复、音频文件名、采样率、嘴型、时间戳）
        self.last_reply: Optional[str] = None
        self.last_wav: Optional[str] = None
        self.last_sr: Optional[int] = None
        self.last_visemes: Optional[Any] = None
        self.last_tts_ts: float = 0.0

        # 频率限制：两次 TTS 合成的最小间隔（秒）
        self.tts_min_interval = float(tts_min_interval_sec)

    def _need_tts(self, reply: str) -> bool:
        """
        判定是否需要重新合成 TTS：
          - 新的 reply 与上次不同；或
          - 距离上次合成已超过最小间隔
        """
        now = time.time()
        if self.last_reply is None:
            return True
        if reply != self.last_reply:
            return True
        if (now - self.last_tts_ts) >= self.tts_min_interval:
            return True
        return False

    def _tts_once(self, reply: str) -> Tuple[str, int, Optional[Any]]:
        """
        执行一次 TTS（写 wav，并生成可选嘴型）。内部会更新缓存。
        PiperTTS 实现中已含失败回退（静音），因此这里不抛异常。
        """
        # 一次性拿到 wav + pcm，避免二次读写
        wav_file, sr, pcm = self.tts.synth_to_wavfile(reply, return_pcm=True)

        # 生成嘴型（失败不影响主流程）
        try:
            vis = viseme_from_audio(pcm, sr)
        except Exception:
            vis = None

        # 写入缓存
        self.last_reply = reply
        self.last_wav = wav_file
        self.last_sr = sr
        self.last_visemes = vis
        self.last_tts_ts = time.time()

        return wav_file, sr, vis

    def step(self, frame_bgr: np.ndarray, user_text: Optional[str] = "") -> Dict[str, Any]:
        """
        输入: 单帧 BGR (numpy)，可选用户文本（M1可空）
        输出: 情绪标签、回复文本、bbox、音频文件名、采样率、置信度、嘴型（可选）
        """
        # 1) FER 推理
        emo_out = self.fer.infer(frame_bgr)  # 期望: {'probs', 'conf', 'bbox'}
        probs = np.asarray(emo_out.get("probs", np.zeros(len(CLASSES), dtype=np.float32)))
        conf = float(emo_out.get("conf", np.max(probs) if probs.size else 0.0))
        bbox = emo_out.get("bbox", None)
        emo_label = argmax_label(probs, CLASSES) if probs.size else "neutral"

        # 2) 文本策略生成回复
        reply = make_reply(user_text or "", emo_label)

        # 3) TTS：只在需要时重新合成；否则复用缓存
        if self._need_tts(reply):
            wav_file, sr, vis = self._tts_once(reply)
        else:
            # 复用缓存（在服务刚启动但未合成过的极端场景，兜底再合成一次）
            if self.last_wav is None or self.last_sr is None:
                wav_file, sr, vis = self._tts_once(reply)
            else:
                wav_file, sr, vis = self.last_wav, self.last_sr, self.last_visemes

        # 4) 返回给前端/上层 API
        return {
            "emo": emo_label,
            "conf": conf,
            "reply": reply,
            "wav": wav_file,   # 前端通过 /audio/{wav} 获取
            "sr": sr,
            "bbox": bbox,
            "visemes": vis,
        }
