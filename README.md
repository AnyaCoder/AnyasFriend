## Introduction

**AnyasFriend** is an intelligent conversational AI framework that integrates voice recognition, speech synthesis, natural language processing, and voice activity detection into a unified system. Designed to be versatile and user-friendly, it adapts to a wide range of conversational needs.

## Features:

- **Multimodal Communication**: Supports seamless voice and text-based interactions.
- **Voice Input**: Converts speech to text using sophisticated ASR technology.
- **Speech Output**: Generates natural, human-like speech with TTS.
- **Smart Responses**: Powered by LLMs to deliver contextually accurate replies.
- **Modular & Scalable**: Customizable to integrate a variety of ASR, VAD, and TTS systems.

## Installation

```bash
conda create -n anya python=3.12
conda activate anya
# core deps
pip install pdm
pdm sync
# heavy deps
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# speaker diarization
pip install pyannote.audio addict datasets fastcluster hdbscan simplejson
```

## Start

```bash
# First, generate config.yaml (For first time)
python -m anyasfriend.config
# Then configure llm,asr... in config.yaml
# Now, start server
python main.py

```

Next, connect micphone stream (16000hz) to ws://localhost:8765 (default)

or use [fish-speech-gui](https://github.com/AnyaCoder/fish-speech-gui/releases) -> Chat Agent -> Settings -> Mic Setting (Always listening)

By doing so, you can chat to your bf/gf now.

## Structure

<details>
  <summary>点击展开查看</summary>
<pre>
AnyasFriend/
│
├── README.md                       # 说明
├── pyproject.toml                  # 轻量级依赖
├── requirements.txt               # 比较重量级的依赖
├── .gitignore
├── pdm.lock                        # 锁定了一些重要依赖
│   
└── anyasfriend/                # 主要的项目代码
    ├── __init__.py
    ├── chatbot.py              # 聊天机器人核心类
    ├── components/             # 组件模块
    │   ├── __init__.py
    │   ├── interfaces/         # 抽象类目录
    │   │   ├── __init__.py
    │   │   ├── asr.py          # 语音识别模块的抽象类
    │   │   ├── vad.py          # 语音活动检测模块的抽象类
    │   │   ├── tts.py          # 语音合成模块的抽象类
    │   │   ├── llm.py          # 大语言模型模块的抽象类
    │   │   ├── memory.py       # 记忆模块的抽象类
    │   │   └── knowledge_base.py  # 知识库模块的抽象类
    │   ├── asr/                # 语音识别模块（不同版本）
    │   ├── vad/                # 语音活动检测模块（不同版本）
    │   ├── tts/                # 语音合成模块（不同版本）
    │   ├── llm/                # 大语言模型模块（不同版本）
    │   ├── memory/             # 记忆模块（不同版本）
    │   ├── media/              # 多媒体播放
    │   └── knowledge_base/     # 知识库模块（不同版本）
    ├── factory.py              # 工厂模式，创建聊天机器人实例
    ├── config.py               # 生成/使用配置文件
    └── utils.py                # 工具函数

</pre>

</details>

## Credits

All modules used in this project.
