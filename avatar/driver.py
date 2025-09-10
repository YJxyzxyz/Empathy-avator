# -*- coding: utf-8 -*-
import numpy as np

def viseme_from_audio(samples: np.ndarray, sr=22050, hop=512):
    """
    简化口型驱动: 用帧能量 -> 嘴巴开合尺度
    前端按 open ∈ [0,1] 做插值即可
    """
    if samples is None or len(samples) == 0:
        return [{"open": 0.0}]
    window = max(1, hop)
    n = max(1, len(samples)//window)
    energy = [float(np.sqrt(np.mean(samples[i*window:(i+1)*window]**2) + 1e-9)) for i in range(n)]
    energy = np.array(energy)
    energy = energy / (energy.max() + 1e-9)
    return [{"open": float(v)} for v in energy.tolist()]
