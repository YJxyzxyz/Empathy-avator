# -*- coding: utf-8 -*-
import numpy as np

def viseme_from_audio(samples: np.ndarray, sr: int, fps: int = 25, eps: float = 1e-8):
    if samples is None or len(samples) == 0:
        return {"fps": int(fps), "energy": []}
    # 统一到 float32，范围 [-1,1]
    if samples.dtype == np.int16:
        x = samples.astype(np.float32) / 32768.0
    else:
        x = samples.astype(np.float32)

    x = x - np.mean(x)
    window = max(1, int(sr / fps))
    n = max(1, len(x) // window)

    energy = []
    for i in range(n):
        seg = x[i*window:(i+1)*window]
        if seg.size == 0:
            energy.append(0.0); continue
        rms = np.sqrt(np.maximum(np.mean(seg * seg), eps))
        energy.append(float(rms))

    e = np.asarray(energy, dtype=np.float32)
    e -= e.min()
    if e.max() > eps:
        e /= (e.max() + eps)
    return {"fps": int(fps), "energy": e.tolist()}
