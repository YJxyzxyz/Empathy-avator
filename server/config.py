# -*- coding: utf-8 -*-
"""Configuration helpers for the Empathy Avatar runtime."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

try:  # Optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - PyYAML may be absent
    yaml = None  # type: ignore


DEFAULTS: Dict[str, Any] = {
    "fer_onnx": "models/fer/weights/fer_mbv3.onnx",
    "piper_exe": "tts/piper.exe",
    "piper_voice": "tts/voices/zh_cn_voice.onnx",
    "audio_tmp_dir": "tmp_audio",
    "audio_sample_rate": 16000,
    "audio_chunk_ms": 480,
    "camera_width": 640,
    "camera_height": 480,
    "camera_fps": 15,
    "camera_indices": (0, 1, 2),
    "tts_min_interval": 2.0,
}

ENV_KEYS: Dict[str, str] = {
    "fer_onnx": "FER_ONNX",
    "piper_exe": "PIPER_EXE",
    "piper_voice": "PIPER_VOICE",
    "audio_tmp_dir": "AUDIO_TMP",
    "audio_sample_rate": "AUDIO_SAMPLE_RATE",
    "audio_chunk_ms": "AUDIO_CHUNK_MS",
    "camera_width": "CAMERA_WIDTH",
    "camera_height": "CAMERA_HEIGHT",
    "camera_fps": "CAMERA_FPS",
    "camera_indices": "CAMERA_INDICES",
    "tts_min_interval": "TTS_MIN_INTERVAL",
}


def _minimal_yaml(text: str) -> Dict[str, Any]:
    """Very small YAML subset parser used when PyYAML is unavailable."""
    data: Dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            data[key] = None
            continue
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            if not inner:
                data[key] = []
            else:
                items = [item.strip().strip("'\"") for item in inner.split(",") if item.strip()]
                data[key] = items
            continue
        if value.lower() in {"true", "false"}:
            data[key] = value.lower() == "true"
            continue
        if value.isdigit():
            data[key] = int(value)
            continue
        try:
            data[key] = float(value)
            continue
        except ValueError:
            pass
        data[key] = value.strip("'\"")
    return data


def _load_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        loaded = yaml.safe_load(text)
        return dict(loaded or {})
    return _minimal_yaml(text)


def _env_overrides(prefix: str) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for field, suffix in ENV_KEYS.items():
        env_key = f"{prefix}{suffix}"
        if env_key in os.environ:
            overrides[field] = os.environ[env_key]
    return overrides


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_path(value: Any, default: str) -> Path:
    path = Path(str(value if value is not None else default)).expanduser()
    return path


def _to_indices(value: Any, default: Tuple[int, ...]) -> Tuple[int, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        parts = [v.strip() for v in value.split(",") if v.strip()]
        if not parts:
            return tuple(default)
        return tuple(_to_int(p, default[0]) for p in parts)
    if isinstance(value, (list, tuple, set)):
        parsed = [
            _to_int(item, default[0])
            for item in value
            if isinstance(item, (int, float, str))
        ]
        return tuple(parsed) if parsed else tuple(default)
    return (_to_int(value, default[0]),)


@dataclass
class AppConfig:
    fer_onnx: Path
    piper_exe: Path
    piper_voice: Path
    audio_tmp_dir: Path
    audio_sample_rate: int
    audio_chunk_ms: int
    camera_width: int
    camera_height: int
    camera_fps: int
    camera_indices: Tuple[int, ...]
    tts_min_interval: float

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "AppConfig":
        data = dict(DEFAULTS)
        for key, value in mapping.items():
            if value is None or key not in data:
                continue
            data[key] = value
        fer_onnx = _to_path(data["fer_onnx"], DEFAULTS["fer_onnx"])
        piper_exe = _to_path(data["piper_exe"], DEFAULTS["piper_exe"])
        piper_voice = _to_path(data["piper_voice"], DEFAULTS["piper_voice"])
        audio_tmp = _to_path(data["audio_tmp_dir"], DEFAULTS["audio_tmp_dir"])
        audio_sr = _to_int(data["audio_sample_rate"], DEFAULTS["audio_sample_rate"])
        audio_chunk = _to_int(data["audio_chunk_ms"], DEFAULTS["audio_chunk_ms"])
        cam_w = _to_int(data["camera_width"], DEFAULTS["camera_width"])
        cam_h = _to_int(data["camera_height"], DEFAULTS["camera_height"])
        cam_fps = _to_int(data["camera_fps"], DEFAULTS["camera_fps"])
        indices = _to_indices(data.get("camera_indices"), DEFAULTS["camera_indices"])
        tts_interval = _to_float(data["tts_min_interval"], DEFAULTS["tts_min_interval"])
        return cls(
            fer_onnx=fer_onnx,
            piper_exe=piper_exe,
            piper_voice=piper_voice,
            audio_tmp_dir=audio_tmp,
            audio_sample_rate=audio_sr,
            audio_chunk_ms=audio_chunk,
            camera_width=cam_w,
            camera_height=cam_h,
            camera_fps=cam_fps,
            camera_indices=indices,
            tts_min_interval=tts_interval,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fer_onnx": str(self.fer_onnx),
            "piper_exe": str(self.piper_exe),
            "piper_voice": str(self.piper_voice),
            "audio_tmp_dir": str(self.audio_tmp_dir),
            "audio_sample_rate": self.audio_sample_rate,
            "audio_chunk_ms": self.audio_chunk_ms,
            "camera_width": self.camera_width,
            "camera_height": self.camera_height,
            "camera_fps": self.camera_fps,
            "camera_indices": list(self.camera_indices),
            "tts_min_interval": self.tts_min_interval,
        }

    def ensure_directories(self) -> None:
        self.audio_tmp_dir.mkdir(parents=True, exist_ok=True)


PathLike = Union[os.PathLike[str], str]


def load_config(path: Optional[PathLike] = None, *, env_prefix: str = "AVATAR_") -> AppConfig:
    config_map: Dict[str, Any] = {}
    search_path = Path(path) if path else Path("configs/default.yaml")
    config_map.update(_load_file(search_path))
    config_map.update(_env_overrides(env_prefix))
    return AppConfig.from_mapping(config_map)
