# anyasfriend/components/interfaces/asr.py

from abc import ABC, abstractmethod


class ASR(ABC):
    @abstractmethod
    async def recognize_speech(self, audio_data: bytes) -> str:
        """
        将音频数据转换为文本。
        :param audio_data: 输入的音频数据
        :return: 转换后的文本
        """
        pass
