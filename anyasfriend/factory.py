# anyasfriend/factory.py

from anyasfriend.chatbot import Chatbot
from anyasfriend.components.asr import FunASR, FunASRConfig
from anyasfriend.components.interfaces import LLMBaseConfig, TTSBaseConfig
from anyasfriend.components.llm import DeepSeekLLM, DeepSeekLLMConfig
from anyasfriend.components.memory import InMemory
from anyasfriend.components.tts import FishTTS, FishTTSConfig
from anyasfriend.components.vad import SileroVAD, SileroVADConfig
from anyasfriend.config import ChatbotConfig


class ChatbotFactory:
    @staticmethod
    def create_chatbot(config: ChatbotConfig) -> Chatbot:
        # 初始化记忆模块
        memory = None
        if config.memory.version == "in-memory":
            memory = InMemory()
        else:
            raise ValueError(f"Unsupported Memory version: {config.memory.version}")

        # 根据配置选择 LLM 版本和 API 配置
        llm = None
        if config.llm.version == "deepseek-chat":
            llm = DeepSeekLLM(
                memory=memory,
                config=DeepSeekLLMConfig(
                    base=LLMBaseConfig(
                        api_key=config.llm.api_key,
                        api_url=config.llm.api_url,
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
                        api_url=config.tts.api_url,
                    )
                )
            )
        else:
            raise ValueError(f"Unsupported TTS version: {config.tts.version}")

        # 根据配置选择 ASR 版本和 API 配置
        asr = None
        if config.asr.version == "funasr":
            asr = FunASR(config=FunASRConfig(disable_update=config.asr.disable_update))
        else:
            raise ValueError(f"Unsupported ASR version: {config.asr.version}")

        # 根据配置选择 VAD 版本和 API 配置
        vad = None
        if config.vad.version == "silero-vad":
            vad = SileroVAD(config=SileroVADConfig())
        else:
            raise ValueError(f"Unsupported VAD version: {config.vad.version}")

        # 创建聊天机器人实例
        chatbot = Chatbot(asr=asr, vad=vad, tts=tts, llm=llm, memory=memory)
        return chatbot
