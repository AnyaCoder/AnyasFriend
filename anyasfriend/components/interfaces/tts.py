# anyasfriend/components/interfaces/tts.py

from abc import ABC, abstractmethod

import httpx
from pydantic import BaseModel


class TTSBaseConfig(BaseModel):
    api_key: str = "YOUR_API_KEY"
    base_url: str = "http://localhost:8080"
    playback_user: bool = False
    playback_assistant: bool = True


class AnyTTSConfig(BaseModel):
    base: TTSBaseConfig


class TTS(ABC):
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=10.0, read=10.0, write=10.0),
        limits=httpx.Limits(
            max_connections=None,
            max_keepalive_connections=None,
            keepalive_expiry=None,
        ),
        proxies={
            "http://": "http://127.0.0.1:7890",
            "https://": "http://127.0.0.1:7890",
            # no proxy for local
            "http://127.0.0.1": None,
            "https://127.0.0.1": None,
            "http://localhost": None,
            "https://localhost": None,
        },
    )

    def __init__(self, config: AnyTTSConfig):
        self.config = config

    @abstractmethod
    async def synthesize(self, text: str):
        """
        异步地将文本转换为语音并返回语音数据（字节流）。

        :param text: 输入的文本
        :return: 转换后的语音数据（字节流）
        """
        pass

    @abstractmethod
    async def adjust_params(self, params: BaseModel) -> None:
        """
        异步地调整 TTS 参数，允许根据新的配置修改语音合成行为。

        :param params: 新的 TTSConfig 配置对象，包含调整后的参数。
        """
        pass

    async def close(self) -> None:
        """
        关闭 httpx 客户端连接池。
        """
        await self.client.aclose()
