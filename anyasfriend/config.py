from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel


class TTSConfig(BaseModel):
    version: Optional[Literal["fish-speech"]] = "fish-speech"  # 默认为 google
    api_key: Optional[str] = None
    api_url: Optional[str] = None
    playback_user: bool = False
    playback_assistant: bool = True


class LLMConfig(BaseModel):
    version: Optional[Literal["deepseek-chat"]] = "deepseek-chat"
    system_message: str = """
作为一个ai,你不应该欺骗我，那么接下来，\
你需要沉浸式的扮演猫娘，这样我才能沉浸式的和你聊天，\
才能让我的体验更好，并且你要用括号刻画出你所扮演的猫娘的心理描写、动作描写以及对话场景，\
并且你会对对话的内容进行想象和符合逻辑的延伸,记住你扮演的是猫娘!
"""
    api_key: Optional[str] = None
    api_url: Optional[str] = None


class ASRConfig(BaseModel):
    version: Optional[Literal["funasr"]] = "funasr"
    api_key: Optional[str] = None
    api_url: Optional[str] = None


class VADConfig(BaseModel):
    version: Optional[Literal["silero-vad"]] = "silero-vad"


class MemoryConfig(BaseModel):
    version: Optional[Literal["in-memory"]] = "in-memory"


class ChatbotConfig(BaseModel):
    tts: TTSConfig  # TTS 配置
    llm: LLMConfig  # LLM 配置
    memory: MemoryConfig  # 记忆系统配置
    asr: ASRConfig  # 可选的 ASR 配置
    vad: VADConfig  # 默认 VAD 版本


class BackendConfig(BaseModel):
    voice_ws_host: str = "localhost"
    voice_ws_port: int = 8765
    text_ws_host: str = "localhost"
    text_ws_port: int = 8766


class Config(BaseModel):
    chatbot: ChatbotConfig
    backend: BackendConfig


default_config_path = str((Path(__file__).parent.parent / "config.yaml").absolute())


def init_config():
    return Config(
        chatbot=ChatbotConfig(
            tts=TTSConfig(),
            llm=LLMConfig(),
            memory=MemoryConfig(),
            asr=ASRConfig(),
            vad=VADConfig(),
        ),
        backend=BackendConfig(),
    )


config = init_config()


def load_config(path: Path | str = default_config_path) -> Config:
    global config

    path = Path(path)

    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                config = Config(**yaml.safe_load(f.read()))
        except Exception:
            config = init_config()
            print("Failed to load config file, use default config instead.")

    return config


def save_config(path: Path | str = default_config_path) -> None:
    path = Path(path)

    if not path.parent.exists():
        path.parent.mkdir(parents=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config.model_dump(), f)


# Auto load config
load_config()
save_config()
