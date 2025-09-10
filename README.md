# Empathy-avator

#### author：YJxyzxyz

<img src="https://github.com/YJxyzxyz/Empathy-avator/blob/master/Empathy-avator.png" width="210px">

数字人+多模态情感识别 构建一个虚拟陪伴机器，通过摄像头输入流分析用户的情感/语言/文字输入，识别情绪状态，然后用NLP/大模型输出对话，驱动数字人来回答

开发板：Jetson nano

python：3.10

项目目录：

```
empathy-avatar/
├─ data/
│  ├─ raw/           # 原始数据（本地放多个公开集）
│  └─ processed/
├─ models/
│  ├─ fer/           # 表情识别（含导出/推理脚本）
│  ├─ ser/           # 语音情感（流式）
│  ├─ nlp/           # 文本情感/意图（小模型+PEFT）
│  └─ fusion/        # 门控融合/Cross-Attention
├─ dialog/
│  ├─ policy.py      # 情绪调节器+安全守护
│  └─ engine.py      # 本地小模型或规则/RAG
├─ tts/
│  └─ synth.py       # Piper TTS 封装
├─ avatar/
│  └─ driver.py      # 2D 口型（viseme）驱动
├─ server/
│  ├─ api.py         # FastAPI + WebSocket
│  └─ pipeline.py    # 实时管线（M1 可运行）
├─ configs/
│  └─ default.yaml   # 路径/权重/超参
└─ scripts/
   ├─ prepare_datasets.py
   └─ evaluate_end2end.py
```

## M1 1.0 

### 最小可运行demo 单机本地、纯离线

视觉 FER + 基础 TTS，控制台打印情绪 → 朗读安慰话术

### 环境搭建

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu12.x
```

```powershell
pip install opencv-python mediapipe numpy pydantic fastapi uvicorn[standard] websockets pyannote.audio librosa sounddevice webrtcvad
pip install onnx onnxruntime  # gpu or cpu version
```

```powershell
pip install piper-tts
```



## M1 1.1

#### ·修复bug:

避免每一帧都触发 **TTS**导致后台不断下载音频 

解决Windows 句柄占用问题，避免文件同时占用

修改嘴型能量计算方式，进行安全裁剪

添加audio自动清理，防止内存爆炸

#### ·新功能:

对 FER 输出做多数投票平滑（最近 8 帧大多数一致才认为“状态稳定”）。只在稳定情绪改变时更新 reply。

TTS 双阈值：文本必须变化 且 距上次合成超过 tts_min_interval 才合成；否则复用上次音频

添加视频流输出以及bbox绘制
