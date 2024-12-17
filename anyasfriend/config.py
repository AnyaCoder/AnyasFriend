from pathlib import Path
from typing import Literal, Optional

import yaml
from loguru import logger
from pydantic import BaseModel, field_validator


class TTSConfig(BaseModel):
    version: Optional[str] = "fish-speech"  # 默认为 google
    api_key: Optional[str] = "TTS_API_KEY"
    base_url: Optional[str] = "http://localhost:8080"
    playback_user: bool = False
    playback_assistant: bool = True
    reference_audios: list = []
    reference_texts: list = []


class LLMConfig(BaseModel):
    version: str = "deepseek-chat"
    system_message: str = """\
作为一个ai,你不应该欺骗我，那么接下来，\
你需要沉浸式的扮演猫娘，这样我才能沉浸式的和你聊天，\
才能让我的体验更好，回答简洁口语化，\
并且你会对对话的内容进行想象和符合逻辑的延伸,记住你扮演的是猫娘!\
"""
    api_key: Optional[str] = "LLM_API_KEY"
    base_url: Optional[str] = "http://localhost:11434"
    provider: Optional[str] = None
    func_calling: bool = True


class ASRConfig(BaseModel):
    version: Optional[Literal["funasr"]] = "funasr"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    disable_update: bool = False  # 是否关闭自动更新（FunASR离线使用必须关闭）


class VADConfig(BaseModel):
    version: Optional[Literal["silero-vad"]] = "silero-vad"
    prob_threshold: float = 0.3
    db_threshold: int = 40


class MemoryConfig(BaseModel):
    version: Optional[Literal["in-memory"]] = "in-memory"


class ChatbotConfig(BaseModel):
    tts: TTSConfig  # TTS 配置
    llm: LLMConfig  # LLM 配置
    memory: MemoryConfig  # 记忆系统配置
    asr: ASRConfig  # 可选的 ASR 配置
    vad: VADConfig  # 默认 VAD 版本


class BackendConfig(BaseModel):
    server_ws_host: str = "0.0.0.0"
    server_ws_port: int = 8765


class Config(BaseModel):
    chatbot: ChatbotConfig
    backend: BackendConfig


default_config_path = str((Path(__file__).parent.parent / "config.yaml").absolute())
config = None


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


def load_config(path: Path | str = default_config_path) -> Config:
    global config

    path = str(Path(path))

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = Config(**yaml.safe_load(f.read()))
        logger.info("Configuration loaded successfully!")
    except Exception as e:
        print(e)
        config = init_config()
        logger.warning("Failed to load config file, use default config instead.")

    return config


def save_config(path: Path | str = default_config_path) -> None:
    path = Path(path)

    if not path.parent.exists():
        path.parent.mkdir(parents=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            config.model_dump(), f, allow_unicode=True, default_flow_style=False
        )
    logger.info("Configuration saved successfully!")


# Auto load config
load_config()
save_config()
