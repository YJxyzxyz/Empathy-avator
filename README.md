# Empathy-avator

数字人+多模态情感识别 构建一个虚拟陪伴机器，通过摄像头输入流分析用户的情感/语言/文字输入，识别情绪状态，然后用NLP/大模型输出对话，驱动数字人来回答

开发板：Jetson nano

python：3.10

项目目录：

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
