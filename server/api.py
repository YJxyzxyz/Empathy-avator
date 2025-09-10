# -*- coding: utf-8 -*-
import os, cv2, asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from server.pipeline import Pipeline

# 路径按需调整
FER_ONNX   = "models/fer/weights/fer_mbv3.onnx"          # 你的 FER ONNX
PIPER_EXE  = "tts/piper.exe"
PIPER_VOICE= "tts/voices/zh_cn_voice.onnx"
AUDIO_TMP  = "tmp_audio"                                  # wav 输出目录

app = FastAPI(title="Empathy Avatar M1 (Local Offline)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# 初始化管线
pipe = Pipeline(
    fer_onnx=FER_ONNX,
    piper_exe=PIPER_EXE,
    piper_voice=PIPER_VOICE,
    audio_tmp_dir=AUDIO_TMP
)

@app.get("/health")
def health():
    ok = os.path.exists(FER_ONNX)
    return {"ok": ok, "fer": FER_ONNX, "piper": os.path.exists(PIPER_EXE)}

@app.get("/audio/{fname}")
def get_audio(fname: str):
    # 提供 wav 下载播放
    fp = Path(AUDIO_TMP) / fname
    if not fp.exists():
        return PlainTextResponse("Not Found", status_code=404)
    return FileResponse(str(fp), media_type="audio/wav")

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    cap = cv2.VideoCapture(0)  # 默认摄像头
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                await asyncio.sleep(0.03)
                continue

            # 这里文本输入留空（M1），后续可扩充从ws接收文本
            result = pipe.step(frame, user_text="")

            # 发给前端：标签、回复、音频文件名、置信度、可选bbox
            await ws.send_json({
                "emo": result["emo"],         # ✅ 改成和 pipeline.py 一致
                "reply": result["reply"],
                "wav": result["wav"],
                "sr": result["sr"],
                "conf": result["conf"],
                "bbox": result.get("bbox"),
                # "visemes": result["visemes"],  # 如需前端做嘴形动画可打开
            })

            await asyncio.sleep(0.06)  # ~16 FPS（降低一点避免占满）
    except WebSocketDisconnect:
        pass
    finally:
        cap.release()
