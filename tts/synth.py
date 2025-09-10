# tts/synth.py
# -*- coding: utf-8 -*-
import os
import subprocess
import tempfile
import wave
import numpy as np
from pathlib import Path
import uuid
import shutil

class PiperTTS:
    def __init__(self, piper_exe: str, voice_path: str, out_dir: str = "tmp_audio", sample_rate: int = 22050):
        self.piper_exe = Path(piper_exe)
        self.voice_path = Path(voice_path)
        self.sample_rate = sample_rate
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _gc_audio_dir(self, keep: int = 50):
        """
        清理输出目录，最多保留 keep 个最近的 wav 文件
        """
        files = sorted(self.out_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files[keep:]:
            try:
                p.unlink()
            except Exception:
                pass

    def synth(self, text: str):
        """
        直接返回 PCM (numpy.int16) 和采样率
        """
        # 创建一个临时文件接收 wav
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpwav:
            tmpwav_path = Path(tmpwav.name)

        try:
            cmd = [
                str(self.piper_exe),
                "-m", str(self.voice_path),
                "-s", str(self.sample_rate),
                "-f", str(tmpwav_path)
            ]
            # Windows 下 text 要通过 stdin 输入
            proc = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            if proc.returncode != 0:
                raise RuntimeError(f"Piper TTS 失败: {proc.stderr.decode(errors='ignore')}")

            # 读取 wav
            with wave.open(str(tmpwav_path), "rb") as w:
                sr = w.getframerate()
                n = w.getnframes()
                pcm = np.frombuffer(w.readframes(n), dtype=np.int16)

            return pcm, sr
        finally:
            # 删除临时文件
            try:
                tmpwav_path.unlink()
            except Exception:
                pass

    def synth_to_wavfile(self, text: str):
        """
        生成 wav 文件，保存到 out_dir 下，返回 (文件名, 采样率)
        """
        fname = f"{uuid.uuid4().hex[:12]}.wav"
        out_path = self.out_dir / fname

        cmd = [
            str(self.piper_exe),
            "-m", str(self.voice_path),
            "-s", str(self.sample_rate),
            "-f", str(out_path)
        ]
        proc = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Piper TTS 失败: {proc.stderr.decode(errors='ignore')}")

        # 自动清理旧的 wav
        self._gc_audio_dir(keep=50)

        return fname, self.sample_rate
