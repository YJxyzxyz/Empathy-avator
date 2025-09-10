# -*- coding: utf-8 -*-
import os
import subprocess
import uuid
import wave
import time
from pathlib import Path
import numpy as np
import sys

SR_DEFAULT = 22050

def _write_wav_int16(path: str, pcm: np.ndarray, sr: int):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.astype(np.int16).tobytes())

def _safe_unlink(path: Path, retries: int = 5, wait_ms: int = 80):
    """Windows 上有时句柄尚未释放，重试删除"""
    for i in range(retries):
        try:
            if path.exists():
                path.unlink()
            return True
        except PermissionError:
            time.sleep(wait_ms / 1000.0)
    return False

class PiperTTS:
    def __init__(self, piper_exe="tts/piper.exe", voice_path="tts/voices/zh_cn_voice.onnx",
                 sample_rate=SR_DEFAULT, out_dir="tmp_audio", espeak_dir="tts/espeak-ng-data"):
        self.exe = piper_exe
        self.voice = voice_path
        self.sr = int(sample_rate)
        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.espeak = espeak_dir  # 可为空；若提供会更稳

        self.ok = Path(self.exe).exists() and Path(self.voice).exists()
        if not self.ok:
            print(f"[TTS] 资源缺失: piper={Path(self.exe).exists()} voice={Path(self.voice).exists()}，将回退静音。")

    def _silence_pcm(self, dur_sec=0.3):
        return (np.zeros(int(self.sr * dur_sec), dtype=np.int16), self.sr)

    def _piper_run(self, text: str, out_path: Path) -> bool:
        """让 piper 直接写出 out_path（注意：此处不提前打开这个文件）"""
        if not self.ok or not text or not text.strip():
            return False
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # 组装命令（尽量少参数，避免不兼容的 -s；若你确认采样率就加 -s）
        cmd = [self.exe, "-m", self.voice, "-f", str(out_path)]
        if self.espeak and Path(self.espeak).exists():
            cmd += ["--espeak_data", self.espeak]
        # 若你已确认模型采样率为 22050，可解开下一行
        # cmd += ["-s", str(self.sr)]

        try:
            p = subprocess.run(
                cmd,
                input=text.encode("utf-8"),  # 关键：UTF-8
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
        except Exception as e:
            print("[TTS] 启动 piper 失败：", e)
            return False

        if p.returncode != 0:
            print("[TTS] piper 退出码:", p.returncode)
            err = p.stderr.decode("utf-8", errors="ignore")
            print("[TTS] stderr:", err[:400])
            return False

        if not out_path.exists():
            print("[TTS] 未生成输出文件:", out_path)
            return False

        size = out_path.stat().st_size
        if size < 64:
            print("[TTS] 输出文件过小(可能空文件):", size, out_path)
            return False

        # 简单检查 RIFF 头
        try:
            with open(out_path, "rb") as f:
                if f.read(4) != b"RIFF":
                    print("[TTS] 非 RIFF WAV:", out_path)
                    return False
        except Exception as e:
            print("[TTS] 读取头失败：", e)
            return False

        return True

    def synth(self, text: str):
        """
        返回 (pcm:int16 ndarray, sr:int)；失败回退静音
        这里使用“真正的临时路径”，但不提前打开文件，避免 Win 句柄冲突
        """
        if not self.ok or not text or not text.strip():
            return self._silence_pcm()

        tmp_path = Path(os.path.join(Path(os.getenv("TEMP", ".")), f"piper_{uuid.uuid4().hex}.wav"))

        ok = self._piper_run(text, tmp_path)
        if not ok:
            _safe_unlink(tmp_path)
            return self._silence_pcm()

        try:
            with wave.open(str(tmp_path), "rb") as w:
                n = w.getnframes()
                sr = w.getframerate()
                frames = w.readframes(n)
            pcm = np.frombuffer(frames, dtype=np.int16)
        except Exception as e:
            print("[TTS] 读取 WAV 失败，回退静音：", e)
            _safe_unlink(tmp_path)
            return self._silence_pcm()

        _safe_unlink(tmp_path)
        return pcm, sr

    def synth_to_wavfile(self, text: str, return_pcm: bool = False):
        """
        在 tmp_audio 写一个随机文件名并返回：
          - return_pcm=False: (fname:str, sr:int)
          - return_pcm=True : (fname:str, sr:int, pcm:np.ndarray)
        任意失败写静音文件，永不抛异常。
        """
        token = uuid.uuid4().hex[:12]
        out_path = self.out_dir / f"{token}.wav"

        if not self.ok or not text or not text.strip():
            pcm, sr = self._silence_pcm()
            _write_wav_int16(out_path, pcm, sr)
            return (out_path.name, sr, pcm) if return_pcm else (out_path.name, sr)

        ok = self._piper_run(text, out_path)
        if not ok:
            pcm, sr = self._silence_pcm()
            _write_wav_int16(out_path, pcm, sr)
            return (out_path.name, sr, pcm) if return_pcm else (out_path.name, sr)

        if return_pcm:
            with wave.open(str(out_path), "rb") as w:
                n = w.getnframes()
                sr = w.getframerate()
                frames = w.readframes(n)
            pcm = np.frombuffer(frames, dtype=np.int16)
            return out_path.name, sr, pcm
        else:
            with wave.open(str(out_path), "rb") as w:
                sr = w.getframerate()
            return out_path.name, sr
