# server/api.py
# -*- coding: utf-8 -*-
import os, cv2, asyncio, time, logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from server.config import AppConfig, load_config
from server.pipeline import Pipeline
from server.audio import SystemAudioStream

# ====== 配置 ======
CONFIG: AppConfig = load_config()
CONFIG.ensure_directories()

FER_ONNX    = str(CONFIG.fer_onnx)
PIPER_EXE   = str(CONFIG.piper_exe)
PIPER_VOICE = str(CONFIG.piper_voice)
AUDIO_TMP   = CONFIG.audio_tmp_dir

# 音频采集配置
AUDIO_SAMPLE_RATE = CONFIG.audio_sample_rate
AUDIO_CHUNK_MS = CONFIG.audio_chunk_ms

# 采集参数
TARGET_W, TARGET_H = CONFIG.camera_width, CONFIG.camera_height
TARGET_FPS = CONFIG.camera_fps
CAM_INDEX_CANDIDATES = list(CONFIG.camera_indices or (0,))  # 若 0 不行，自动尝试其它编号

# ====== FastAPI & 日志 ======
app = FastAPI(title="Empathy Avatar M3 (Local Offline)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)
log = logging.getLogger("uvicorn.error")
log.info("[Config] loaded: %s", CONFIG.to_dict())

# ====== 初始化管线 ======
pipe = Pipeline(
    fer_onnx=FER_ONNX,
    piper_exe=PIPER_EXE,
    piper_voice=PIPER_VOICE,
    audio_tmp_dir=str(AUDIO_TMP),
    tts_min_interval_sec=CONFIG.tts_min_interval,
    ser_sample_rate=AUDIO_SAMPLE_RATE,
)

_audio_candidate = SystemAudioStream(
    sample_rate=AUDIO_SAMPLE_RATE,
    channels=1,
    chunk_ms=AUDIO_CHUNK_MS,
)
AUDIO_STREAM = _audio_candidate if _audio_candidate.available else None

# ====== 全局：摄像头与最新帧缓存 ======
CAP = None
CAP_TASK = None
LATEST_FRAME = None              # numpy BGR
FRAME_LOCK = asyncio.Lock()      # 保护 LATEST_FRAME

# ====== 全局：叠加信息（bbox / emo / conf），供 /video 绘制 ======
OVERLAY_LOCK = asyncio.Lock()
LAST_OVERLAY: Dict[str, Any] = {
    "bbox": None,        # [x, y, w, h] 或 [x1,y1,x2,y2] 或 归一化（0~1）
    "emo": None,         # 'happy' etc.
    "conf": None,        # float
    "ts": 0.0            # 更新时间戳
}
OVERLAY_TTL = 0.6  # 秒；超过不再绘制（避免“拖影”）

# 文本输入缓存
USER_TEXT_LOCK = asyncio.Lock()
LATEST_USER_TEXT: str = ""
LATEST_USER_TEXT_TS: float = 0.0

# ================= 帮助函数 =================

def _open_capture() -> Optional[cv2.VideoCapture]:
    indices = CAM_INDEX_CANDIDATES or [0]
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    for idx in indices:
        cap = cv2.VideoCapture(idx, backend)
        if not cap or not cap.isOpened():
            if cap:
                cap.release()
            # 回退到默认后端
            cap = cv2.VideoCapture(idx)
            if not cap or not cap.isOpened():
                if cap:
                    cap.release()
                continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

        ok, frame = cap.read()
        if ok and frame is not None:
            log.info(
                "[Camera] opened at index %s (%sx%s)", idx, frame.shape[1], frame.shape[0]
            )
            return cap

        cap.release()
    log.error("[Camera] failed to open any camera device (tried %s)", indices)
    return None

def _normalize_bbox_to_xyxy(frame_shape, bbox):
    """
    将传入 bbox 统一为像素级 [x1,y1,x2,y2]，并裁剪到帧内。
    支持：
      - [x,y,w,h]（像素）
      - [x1,y1,x2,y2]（像素）
      - 以上两种的归一化版本（0~1）
    """
    if bbox is None:
        return None
    h, w = frame_shape[:2]

    b = list(bbox)
    if len(b) != 4:
        return None

    # 判断是否归一化（0~1）
    is_norm = all(0.0 <= float(v) <= 1.0 for v in b)
    if is_norm:
        # 先转像素
        b = [float(b[0]) * w, float(b[1]) * h, float(b[2]) * (w if b[2] <= 1 else 1), float(b[3]) * (h if b[3] <= 1 else 1)]

    x, y, a, b2 = map(float, b)

    # 判断是 xyxy 还是 xywh：如果第三个数小于第一个，视为 x2；反之视为 w
    if a > x and b2 > y and (a > 1 or b2 > 1):  # 保守判断：更像 xyxy
        x1, y1, x2, y2 = x, y, a, b2
    else:
        # 视为 xywh
        x1, y1, x2, y2 = x, y, x + a, y + b2

    # 裁剪到画面
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(0, min(w - 1, int(round(x2))))
    y2 = max(0, min(h - 1, int(round(y2))))

    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]

def _draw_overlay(frame: np.ndarray, overlay: Dict[str, Any]) -> np.ndarray:
    """
    在 frame 上叠加 bbox 和情绪文本（就地绘制）。
    """
    if frame is None:
        return frame
    now = time.time()
    ts = overlay.get("ts", 0.0)
    if now - ts > OVERLAY_TTL:
        return frame

    bbox = overlay.get("bbox")
    emo  = overlay.get("emo")
    conf = overlay.get("conf")

    xyxy = _normalize_bbox_to_xyxy(frame.shape, bbox)
    if xyxy is not None:
        x1, y1, x2, y2 = xyxy
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 196, 255), 2, cv2.LINE_AA)

        label = emo if emo else ""
        if conf is not None:
            try:
                label += f" {float(conf):.2f}"
            except Exception:
                pass

        if label:
            # 背景条
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            bx2 = min(frame.shape[1] - 1, x1 + tw + 12)
            by2 = y1 + th + 10
            cv2.rectangle(frame, (x1, y1 - th - 12), (bx2, by2 - th), (0, 196, 255), -1)
            cv2.putText(frame, label, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    return frame

# ================= 采集后台任务 =================

async def _capture_loop():
    global CAP, LATEST_FRAME
    CAP = _open_capture()
    if CAP is None:
        return

    min_interval = 1.0 / max(1, TARGET_FPS)

    try:
        while True:
            start = time.time()
            ok, frame = CAP.read()
            if ok and frame is not None:
                async with FRAME_LOCK:
                    LATEST_FRAME = frame
            elapsed = time.time() - start
            await asyncio.sleep(max(0, min_interval - elapsed))
    except asyncio.CancelledError:
        log.info("[Camera] capture loop cancelled")
    except Exception as e:
        log.exception("[Camera] capture loop error: %s", e)
    finally:
        if CAP is not None:
            CAP.release()
            log.info("[Camera] released")
        CAP = None

# ================= 生命周期钩子 =================

@app.on_event("startup")
async def on_startup():
    global CAP_TASK
    if CAP_TASK is None or CAP_TASK.done():
        CAP_TASK = asyncio.create_task(_capture_loop())
        log.info("[Startup] capture loop started")
    else:
        log.info("[Startup] capture loop already running")

    if AUDIO_STREAM is not None:
        try:
            if AUDIO_STREAM.start():
                log.info("[Startup] audio stream started at %s Hz", AUDIO_SAMPLE_RATE)
            else:
                log.warning("[Startup] audio stream already running or unavailable")
        except Exception:
            log.exception("[Startup] failed to start audio stream")
    else:
        log.warning("[Startup] audio capture unavailable (sounddevice missing?)")


@app.on_event("shutdown")
async def on_shutdown():
    global CAP_TASK
    if CAP_TASK is not None:
        CAP_TASK.cancel()
        try:
            await CAP_TASK
        except Exception:
            pass
        CAP_TASK = None
        log.info("[Shutdown] capture loop stopped")

    if AUDIO_STREAM is not None:
        try:
            AUDIO_STREAM.stop()
            log.info("[Shutdown] audio stream stopped")
        except Exception:
            log.exception("[Shutdown] failed to stop audio stream")

# ================= 健康检查 & 静态资源 =================

@app.get("/health")
def health():
    tmp = AUDIO_TMP
    n_wav = len(list(tmp.glob("*.wav"))) if tmp.exists() else 0
    audio_ready = False
    if AUDIO_STREAM is not None:
        try:
            audio_ready = AUDIO_STREAM.peek_latest() is not None
        except Exception:
            audio_ready = False
    fer_exists = Path(FER_ONNX).exists()
    piper_exists = Path(PIPER_EXE).exists()
    voice_exists = Path(PIPER_VOICE).exists()
    return {
        "ok": fer_exists and piper_exists and voice_exists,
        "fer": FER_ONNX,
        "piper": piper_exists,
        "piper_voice": voice_exists,
        "wav_files": n_wav,
        "video_size": [TARGET_W, TARGET_H],
        "fps_target": TARGET_FPS,
        "camera_opened": CAP is not None,
        "audio_stream": AUDIO_STREAM is not None,
        "audio_has_buffer": audio_ready,
        "audio_sample_rate": AUDIO_SAMPLE_RATE,
        "audio_chunk_ms": AUDIO_CHUNK_MS,
        "config": CONFIG.to_dict(),
    }


@app.get("/config")
def get_config():
    return CONFIG.to_dict()

@app.get("/audio/{fname}")
def get_audio(fname: str):
    fp = AUDIO_TMP / fname
    if not fp.exists():
        return PlainTextResponse("Not Found", status_code=404)
    return FileResponse(str(fp), media_type="audio/wav")


class TextPayload(BaseModel):
    text: str = ""


@app.post("/user-text")
async def set_user_text(payload: TextPayload):
    global LATEST_USER_TEXT, LATEST_USER_TEXT_TS
    txt = payload.text or ""
    async with USER_TEXT_LOCK:
        LATEST_USER_TEXT = txt
        LATEST_USER_TEXT_TS = time.time()
    return {"ok": True, "text": txt, "ts": LATEST_USER_TEXT_TS}


@app.get("/user-text")
async def get_user_text():
    async with USER_TEXT_LOCK:
        return {"text": LATEST_USER_TEXT, "ts": LATEST_USER_TEXT_TS}

# 单帧调试（会叠加 bbox/情绪）
@app.get("/frame")
def get_frame():
    frame = None
    if LATEST_FRAME is not None:
        frame = LATEST_FRAME.copy()

    if frame is None:
        canvas = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
        cv2.putText(canvas, "NO FRAME", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        frame = canvas

    # 叠加 overlay
    try:
        ol = None
        if OVERLAY_LOCK:
            # 这里不是异步环境，可直接读取；如担心竞态，可省略锁
            ol = LAST_OVERLAY.copy()
        if ol:
            frame = _draw_overlay(frame, ol)
    except Exception:
        pass

    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return PlainTextResponse("EncodeError", status_code=500)
    return Response(content=jpg.tobytes(), media_type="image/jpeg")

# ================= 视频流（MJPEG，带叠加） =================

@app.get("/video")
def video_stream():
    boundary = "frame"

    async def agen():
        blank = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
        while True:
            # 拿当前帧
            async with FRAME_LOCK:
                frame = LATEST_FRAME.copy() if LATEST_FRAME is not None else blank.copy()

            # 拿叠加信息并绘制
            try:
                async with OVERLAY_LOCK:
                    ol = LAST_OVERLAY.copy()
                frame = _draw_overlay(frame, ol)
            except Exception:
                pass

            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                await asyncio.sleep(0.05)
                continue

            bytes_ = jpg.tobytes()
            chunk = (
                b"--" + boundary.encode() + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(bytes_)).encode() + b"\r\n\r\n"
            ) + bytes_ + b"\r\n"

            yield chunk
            await asyncio.sleep(1.0 / max(1, TARGET_FPS))

    return StreamingResponse(
        agen(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}"
    )

@app.head("/video")
def video_head():
    return Response(status_code=200, media_type="multipart/x-mixed-replace")

# ================= WebSocket：推送情绪/文本/音频 =================

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            # 取最新帧（供 pipeline 使用）
            async with FRAME_LOCK:
                frame = LATEST_FRAME.copy() if LATEST_FRAME is not None else None

            if frame is None:
                await asyncio.sleep(0.05)
                await ws.send_json({
                    "emo": "neutral",
                    "reply": "初始化摄像头中…",
                    "wav": None,
                    "sr": 22050,
                    "conf": 0.0
                })
                continue

            audio_chunk = None
            if AUDIO_STREAM is not None:
                try:
                    audio_chunk = AUDIO_STREAM.pop_latest()
                except Exception:
                    audio_chunk = None

            user_text_ts = 0.0
            async with USER_TEXT_LOCK:
                user_text = LATEST_USER_TEXT
                user_text_ts = LATEST_USER_TEXT_TS

            result = pipe.step(
                frame,
                audio_chunk=audio_chunk,
                audio_sr=AUDIO_SAMPLE_RATE,
                user_text=user_text,
                user_text_ts=user_text_ts,
            )

            # 更新叠加信息（bbox/emo/conf），供 /video 使用
            try:
                async with OVERLAY_LOCK:
                    LAST_OVERLAY["bbox"] = result.get("bbox")
                    LAST_OVERLAY["emo"]  = result.get("emo")
                    LAST_OVERLAY["conf"] = result.get("conf")
                    LAST_OVERLAY["ts"]   = time.time()
            except Exception:
                pass

            # 发送给前端
            await ws.send_json({
                "emo": result["emo"],
                "reply": result["reply"],
                "wav": result["wav"],
                "sr": result["sr"],
                "conf": result["conf"],
                "modalities": result.get("modalities"),
                "fusion": result.get("fusion"),
                "user_text": user_text,
                "dialog": result.get("dialog"),
                "turn_id": result.get("turn_id"),
                "visemes": result.get("visemes"),
            })

            await asyncio.sleep(0.06)  # ~16 FPS
    except WebSocketDisconnect:
        pass
