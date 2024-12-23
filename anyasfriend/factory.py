# anyasfriend/factory.py

from anyasfriend.components.asr import FunASR, FunASRConfig
from anyasfriend.components.interfaces import LLMBaseConfig, TTSBaseConfig
from anyasfriend.components.llm import (
    DeepSeekLLM,
    DeepSeekLLMConfig,
    OllamaLLM,
    OllamaLLMConfig,
)
from anyasfriend.components.memory import InMemory
from anyasfriend.components.tts import EdgeTTS, EdgeTTSConfig, FishTTS, FishTTSConfig
from anyasfriend.components.vad import SileroVAD, SileroVADConfig
from anyasfriend.config import ChatbotConfig
from anyasfriend.multi_chatbot import Chatbot


class ChatbotFactory:
    @staticmethod
    def create_chatbot(config: ChatbotConfig) -> Chatbot:

        # 根据配置选择 LLM 版本和 API 配置
        llm = None
        if config.llm.provider == "ollama":
            llm = OllamaLLM(
                config=OllamaLLMConfig(
                    base=LLMBaseConfig(
                        api_key=config.llm.api_key,
                        base_url=config.llm.base_url,
                    ),
                    request=dict(
                        model=config.llm.version,
                        stream=True,
                    ),
                ),
            )
        elif config.llm.version == "deepseek-chat":
            llm = DeepSeekLLM(
                config=DeepSeekLLMConfig(
                    base=LLMBaseConfig(
                        api_key=config.llm.api_key,
                        base_url=config.llm.base_url,
                    )
                ),
            )
        else:
            raise ValueError(f"Unsupported LLM version: {config.llm.version}")

        # 根据配置选择 TTS 版本和 API 配置
        tts = None
        if config.tts.version == "fish-speech":
            tts = FishTTS(
                config=FishTTSConfig(
                    base=TTSBaseConfig(
                        api_key=config.tts.api_key,
                        base_url=config.tts.base_url,
                        playback_sample_rate=44100,
                    )
                )
            )
        elif config.tts.version == "edge-tts":
            tts = EdgeTTS(
                config=EdgeTTSConfig(
                    base=TTSBaseConfig(
                        api_key=config.tts.api_key,
                        base_url=config.tts.base_url,
                        playback_sample_rate=24000,
                    )
                )
            )
        else:
            raise ValueError(f"Unsupported TTS version: {config.tts.version}")

        # 根据配置选择 ASR 版本和 API 配置
        if config.asr.version is None:
            asr = None
        elif config.asr.version == "funasr":
            asr = FunASR(config=FunASRConfig(disable_update=config.asr.disable_update))
        else:
            raise ValueError(f"Unsupported ASR version: {config.asr.version}")

        # 根据配置选择 VAD 版本和 API 配置
        vad = None
        if config.vad.version == "silero-vad":
            vad = SileroVAD(
                config=SileroVADConfig(
                    prob_threshold=config.vad.prob_threshold,
                    db_threshold=config.vad.db_threshold,
                    required_hits=config.vad.required_hits,
                    required_misses=config.vad.required_misses,
                    smoothing_window=config.vad.smoothing_window,
                )
            )
        else:
            raise ValueError(f"Unsupported VAD version: {config.vad.version}")

        # 创建聊天机器人实例
        chatbot = Chatbot(asr=asr, vad=vad, tts=tts, llm=llm, memory_cls=InMemory)
        return chatbot
