<img src="https://github.com/YJxyzxyz/Empathy-avator/blob/master/Empathy-avator.png" width="220px">

<h1>Empathy-Avatar</h1>

<strong>一个由实时多模态情感识别驱动的虚拟伴侣。</strong>

<img src="https://img.shields.io/badge/Python-3.10-blue.svg?style=for-the-badge&logo=python" alt="Python 3.10">

<img src="https://img.shields.io/badge/PyTorch-AI-orange.svg?style=for-the-badge&logo=pytorch" alt="PyTorch">

<img src="https://img.shields.io/badge/FastAPI-Server-green.svg?style=for-the-badge&logo=fastapi" alt="FastAPI">

<img src="https://img.shields.io/badge/Platform-Jetson_Nano-yellow.svg?style=for-the-badge&logo=nvidia" alt="Jetson Nano">

> `Empathy-Avatar` 项目旨在打造一个富有同情心的虚拟伴侣。它通过设备的摄像头进行实时的多模态分析，解读用户的面部表情、语音语调和文本内容，从而理解其情绪状态。基于这份理解，项目利用 NLP 技术以及大模型生成共情对话，并驱动一个 2D 数字人化身，提供有温度的互动与支持。

## ✨ 核心功能

- **🧠 多模态情感识别**:
  - **FER**: 实时面部表情识别，捕捉视觉情绪信号。
  - **SER**: 流式语音情感识别，分析声音中的情绪起伏。
  - **NLP**: 文本情感与意图分析，理解语言层面的深层含义。
- **🗣️ 动态对话引擎**:
  - 集成情绪调节与安全守护策略，确保互动积极健康。
  - 支持本地小模型、规则或 RAG 等多种方式生成对话。
- **🤖 生动的 2D 形象**:
  - 通过 `viseme`（口型）精确驱动，实现逼真的唇形同步动画。
- **⚡ 实时与离线**:
  - 基于 `FastAPI` + `WebSocket` 构建高性能后端，实现低延迟交互。
  - 专为 Jetson Nano 等边缘设备优化，可完全离线运行。
- **👍 智能平滑处理**:
  - 对最近 8 帧的识别结果进行多数投票，只有在情绪状态稳定改变时才更新回应，避免频繁波动。
- **🔊 高效 TTS 合成**:
  - 采用双重阈值策略：仅当回复文本变化且距离上次合成超过设定间隔时，才触发新的语音合成，有效降低资源消耗。



## 🛠️ 技术栈

- **核心框架**: Python 3.10, PyTorch
- **AI / ML**: ONNX Runtime, MediaPipe, OpenCV, `pyannote.audio`, Librosa
- **Web & API**: FastAPI, Uvicorn, WebSockets
- **音频处理**: Piper TTS, SoundDevice, webrtcvad

## 🗂️ 项目结构

```
empathy-avatar/
├─ data/
│  ├─ raw/           # 原始数据（本地放多个公开集）
│  └─ processed/     # 经过预处理的数据
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
│  ├─ api.py         # FastAPI + WebSocket 接口
│  └─ pipeline.py    # 实时管线（M1 可运行）
├─ configs/
│  └─ default.yaml   # 路径/权重/超参
└─ scripts/
   ├─ prepare_datasets.py
   └─ evaluate_end2end.py
```



## 🚀 快速开始

本指南将帮助您在本地设备上配置并运行一个最小化的离线演示。

### **环境要求**

- **硬件**: 一台带摄像头的电脑 (推荐 Jetson Nano)。
- **Python**: `3.10` 版本。

### **安装步骤**

1. **克隆代码仓库**

   Bash

   ```
   git clone https://github.com/YJxyzxyz/Empathy-avator.git
   cd Empathy-avator
   ```

2. **安装 PyTorch**

   - 请根据您的 CUDA 版本或选择 CPU 版本进行安装。

   PowerShell

   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu12.x
   ```

3. **安装核心依赖**

   PowerShell

   ```
   pip install opencv-python mediapipe numpy pydantic fastapi uvicorn[standard] websockets pyannote.audio librosa sounddevice webrtcvad
   ```

4. **安装 ONNX Runtime**

   - 根据您的硬件选择 GPU 或 CPU 版本。

   PowerShell

   ```
   # CPU 版本
   pip install onnxruntime
   
   # GPU 版本 (请确保 CUDA/cuDNN 版本匹配)
   pip install onnxruntime-gpu
   ```

5. **安装 TTS 引擎**

   PowerShell

   ```
   pip install piper-tts
   ```



## 📈 更新日志

### **Milestone 3.0**

#### **🚀 新功能:**

- **检索式对话策略**：新增本地 `DialogEngine`，基于关键词加权的轻量级 RAG，结合情绪标签与历史上下文生成更自然的共情回复。触发安全词时自动切换应急策略。
- **数字人嘴型同步**：TTS 输出同时返回 viseme 序列（能量轨迹 + FPS），前端数字人可精准驱动口型动画。
- **对话仪表盘**：Web 客户端新增数字人面板、策略来源、触发原因、检索线索以及对话时间轴，方便观察策略演化。

#### **🧼 体验优化:**

- **情绪/文本联合触发**：情绪稳定发生变化或用户输入更新时才刷新回复，并智能节流 TTS，避免重复合成。
- **会话留存**：后端缓存对话历史、轮次编号与 TTS turn-id，前端滚动列表按顺序展示来访者与数字人的轮次。

### **Milestone 1.1**

#### **🚀 新功能:**

- **情绪平滑**: 对 FER 输出增加了多数投票平滑机制（最近8帧），只有当情绪状态稳定变化时才更新回复。
- **TTS 优化**: 引入双重阈值，只有当文本变化且距上次合成超过最小时间间隔时才重新合成，否则复用上次的音频。
- **视频流输出**: 增加了处理后的视频流输出，并绘制了人脸检测框（bbox）。

#### **🔧 Bug 修复:**

- 修复了每一帧都可能触发 TTS 导致音频重复请求的问题。
- 解决了 Windows 系统下文件句柄被持续占用的问题。
- 改进了嘴型能量计算方式，并增加了安全裁剪。
- 添加了音频文件的自动清理机制，防止内存溢出。

### **Milestone 2.0**

#### **🚀 新功能:**

- **三模态情绪识别**：新增流式语音情感识别（SER）与文本情绪分析，采用启发式特征（RMS/ZCR/频谱质心、关键词词典）实现轻量级推理。
- **规则式融合**：引入按置信度门控的加权融合器，将视觉、语音、文本概率统一输出稳定情绪标签。
- **输入交互增强**：前端增加文本输入与实时状态面板，可查看各模态情绪、融合权重以及风险提示。
- **音频采集管线**：后端集成麦克风采集（sounddevice，可选），WebSocket 推流中同步返回多模态详情。

#### **🧼 体验优化:**

- 新增 `/user-text` API 用于文本同步，Web 客户端支持一键发送/清空输入。
- Web UI 更新为 M2 主题样式，提供多模态分析概览与融合摘要。

### **Milestone 1.0**

- 实现了首个最小可运行的本地离线 Demo。
- 功能包含：视觉 FER + 基础 TTS，在控制台打印识别出的情绪，并朗读预设的安慰话术。
