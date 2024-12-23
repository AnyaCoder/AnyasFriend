<div align="center">
<h1>AnyasFriend</h1>
</div>

<div align="center">
    <a target="_blank" href="https://hub.docker.com/r/anyacoder/anyasfriend">
        <img alt="Docker" src="https://img.shields.io/docker/pulls/anyacoder/anyasfriend?style=flat-square&logo=docker"/>
    </a>
</div>

## Introduction

**AnyasFriend** （interpreted as any-as-friend) is an intelligent conversational AI framework that integrates voice recognition, speech synthesis, natural language processing, and voice activity detection into a unified system. Designed to be versatile and user-friendly, it adapts to a wide range of conversational needs.

此项目处于早期版本，功能会持续更新。

## Features 

Welcome to the **Real-Time Voice Boyfriend/Girlfriend Chat** system, where you can chat and interact with a virtual boyfriend or girlfriend through voice in a fun and natural way. This system combines multiple AI technologies to make the experience as lifelike and engaging as possible. Whether you're looking for companionship, help with tasks, or just a fun conversation, your virtual partner is here for you, 24/7!

------

### Key Features 

#### 1. **Real-Time Conversations** | **实时对话**

- Communicate with your virtual boyfriend or girlfriend through voice in **real time**.
- Whether you speak or type, your partner will listen and respond almost instantly.

#### 2. **Voice Recognition** | **语音识别**

- When you talk, the system **understands** what you're saying and responds accordingly.
- No more typing—just speak naturally, and your virtual partner will react to your words.

#### 3. **Speech Responses** | **语音回答**

- Your virtual partner doesn't just text back, they **talk** to you!
- Their voice is clear and natural, making your interaction feel more personal.

#### 4. **Intelligent Memory** | **智能记忆**

- Your virtual partner **remembers** past conversations and personal details to make future chats more meaningful.
- They’ll bring up past topics, ask about your day, or remind you of fun memories.

#### 5. **Personalized Commands** | **个性化命令**

- You can give commands like "**/clear**" if you want to reset everything or "**/history**" to see what you’ve talked about before.
- The system responds to simple, friendly commands to keep your conversations organized.

#### 6. **Emotion and Personality** | **情感和个性**

- Your virtual boyfriend/girlfriend can express emotions, like happiness, excitement, or sadness, depending on the mood of your conversation.

#### 7. **Flexible Communication** | **灵活交流**

- Whether you're talking to your virtual partner via voice or typing, you can easily switch between the two modes. No need to choose—just talk and type as you wish. Engage with your virtual boyfriend/girlfriend in a way that feels personal and alive.
- Ask them to share jokes, offer advice, or even tell you about their (virtual) day. They’re always ready to chat!

#### 8. **Seamless Connection** | **无缝连接**

- The system allows for smooth, continuous conversation. No interruptions, just flowing dialogue between you and your virtual partner.
- Enjoy a natural back-and-forth without long pauses or delays.



## 先决条件

### Windows User

1. 安装cuda toolkit 12及以上。
2. 安装cudnn 9及以上。
3. 下载miniconda或者python=3.12.

### Linux User (Ubuntu / debian)

```bash
sudo apt-get update
sudo apt-get install \
	ca-certificates libsox-dev build-essential \
    cmake libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg
```

## MacOS User

```bash
brew update
brew install \
  ca-certificates \
  sox \
  cmake \
  portaudio \
  ffmpeg
```



## Installation via CLI

### Windows / Linux User (Ubuntu / debian)

```bash
# after -n can be changed to any other name you like
conda create -n anya312 python=3.12
conda activate anya312
# core deps
pip install -e .
# heavy deps
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

# speaker diarization (deprecated now)
pip install pyannote.audio addict datasets fastcluster hdbscan simplejson
```

### MacOS

```bash
# after -n can be changed to any other name you like
conda create -n anya312 python=3.12
conda activate anya312
# core deps
pip install -e .
# heavy deps (cpu/mps)
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

# speaker diarization (deprecated now)
pip install pyannote.audio addict datasets fastcluster hdbscan simplejson
```

### Docker (Ubuntu)

```bash
# add remote nvidia repo
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
# install nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
# Method 1: from remote
docker pull anyacoder/anyasfriend:latest
# Method 2: from local
# docker build . -t anyasfriend
# wsl2 or linux
docker run -d --gpus all -p 8765:8765 -v /path/to/config.yaml:/opt/anyasfriend/config.yaml -v /path/to/ref_folder:/opt/ref anyacoder/anyasfriend:latest
# see logs in docker
docker logs -f --tail 100 <container id>
```



##  Installation via scripts

### Windows

1. 下载该项目，如果是压缩包，请解压到一个合适的位置。
2. 双击项目文件夹，运行其中的`install_env.bat`，等待依赖安装完成。
3. 点击`start.bat`运行服务。



## Start

### 1. Configure yaml

要使用本项目，使得更好地满足我们的需求，我们需要配置自己所需的模块。

示例配置都写到了`example_config.yaml`里面，但不能直接使用。

我们需要复制一份`example_config.yaml`，在同一个目录下粘贴，并改名为`config.yaml`, 此时才能让项目识别到配置文件。

之后修改`config.yaml`文件即可修改启动配置。下面不妨再把它贴出来，做一个详细的注解：

```yaml
# 后端websocket服务配置项，一般是不用动的
backend:
  server_ws_host: 0.0.0.0
  server_ws_port: 8765
# 主要配置
chatbot:
  # 语音转文字
  asr:
    # 暂时没做网络版的ASR，不用填key和url
    api_key: YOUR_API_KEY
    base_url: null
    # 第一次使用，需要设置disable_update=False下载模型，后续可设置为true
    disable_update: true
    # funasr的SenseVoiceSmall非常够用了
    version: funasr
  # 大语言模型
  llm:
    api_key: YOUR_API_KEY
    # deepseek-chat:  https://api.deepseek.com
    #   ollama:  http://localhost:11434
    base_url: https://api.deepseek.com
    # 工具调用 （设置情绪, 暂时只支持）
    func_calling: true
    # 不用 ollama 的时候设置 `provider` 为 null, 用 ollama 则设为 ollama
    provider: null
    # system_prompt，系统提示词。在客户端里设置。
    # ollama: qwen2.5:3b
    # ollama以外的大模型都需要互联网连接
    version: deepseek-chat
  # 记忆模块（目前比较平凡）
  memory:
    version: in-memory
  # 文本转语音
  tts:
    api_key: YOUR_API_KEY
    # 若使用docker访问宿主机的8080需配置为: http://host.docker.internal:8080
    # fish-speech常用url: http://localhost:8080（本地）, https://api.fish.audio（在线）
    base_url: http://localhost:8080
    # 下面两个弃用了
    playback_assistant: true
    playback_user: false
    # 一些tts模型需要参考音频来复刻音色，需要把他们的音频和文本分别存储, 如果没有就填 []，就像下面的:
    # reference_audios: [], 否则有多少音频就得先打一个减号 -，空格，再接路径。
    reference_audios:
      - D:\voices\example_1.wav
      - D:\voices\example_2.wav
    reference_texts:
      - D:\voices\example_1.lab
      - D:\voices\example_2.lab
    # 目前实现：edge-tts, fish-speech
    version: fish-speech
  # 语音活动检测
  vad:
    db_threshold: 40
    prob_threshold: 0.4
    required_hits: 3
    required_misses: 24
    smoothing_window: 5
    version: silero-vad

```
### 2. Start server

```bash
python main.py
```

正常情况下会出现`ws://0.0.0.0:8765`的信息，表明服务端已经正常启动。

### 3. Download your boy/girl friend!

(Currently can work on windows)

https://github.com/AnyaCoder/anyasfriend/releases

选择最新版本。下载好以后，丢到一个文件夹里。再把下载的resources（live2d)压缩包解压到当前文件夹。

### 4. Wake him/her up!

Double click `anyalive2d.exe`, Then he/she shows up.

双击这个exe，TA就出现啦。

### 5. Interaction

和TA交互，目前三种方式。

	1. 点击ta，ta会做出反应，摆个pose，做个表情。
	2. 右键ta，会跳出一个小菜单，点击`Input text`可以输入文本对话。
	3. 右键ta，点击菜单里的`Audio options`, 然后再将`Audio record`点击勾选上，此时就可以进行实时语音对话状态了。

ta说话的过程中也可被你随时打断。ta会停住，直到你说完

## Future

开发者会持续更新功能。敬请期待。尽量让ta更有活力，更为主动！

## Credits

All modules used in this project.
