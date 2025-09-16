# server/audio.py
# -*- coding: utf-8 -*-
"""Audio capture helper for streaming SER."""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - optional dependency
    sd = None  # type: ignore


class SystemAudioStream:
    """Background microphone capture using sounddevice."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_ms: int = 400,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.chunk_ms = int(chunk_ms)
        self.chunk_samples = max(1, int(self.sample_rate * self.chunk_ms / 1000))
        self._latest: Optional[np.ndarray] = None
        self._latest_ts: float = 0.0
        self._lock = threading.Lock()
        self._stream: Optional["sd.InputStream"] = None
        self._running = False
        self._available = sd is not None

    # ---------------------------------------------------------------- control
    @property
    def available(self) -> bool:
        return bool(self._available)

    def start(self) -> bool:
        if not self.available or self._running:
            return False

        def _callback(indata, frames, time_info, status):  # pragma: no cover - real-time callback
            if status:
                # Drop frames on overflow/underflow but keep running
                pass
            data = np.asarray(indata).reshape(-1)
            with self._lock:
                self._latest = data.copy()
                self._latest_ts = time.time()

        self._stream = sd.InputStream(
            channels=self.channels,
            samplerate=self.sample_rate,
            blocksize=self.chunk_samples,
            callback=_callback,
            dtype="float32",
        )
        self._stream.start()
        self._running = True
        return True

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        self._stream = None
        self._running = False

    # ----------------------------------------------------------------- data api
    def pop_latest(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest is None:
                return None
            data = self._latest.copy()
            self._latest = None
            return data

    def peek_latest(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def last_timestamp(self) -> float:
        with self._lock:
            return float(self._latest_ts)
