from abc import ABC, abstractmethod

import numpy as np


class SpeakerRecognition(ABC):
    """
    抽象类：用于说话人识别的基类
    """

    @abstractmethod
    def extract_features(self, audio_np: np.ndarray) -> int:
        """
        提取音频特征，用于后续的音色比对。
        :param audio_path: 音频文件路径或内存中的音频数据
        :return: 音频特征的聚类编号
        """
        pass

    @abstractmethod
    def recognize(self, audio_np: np.ndarray) -> bool:
        """
        调用完成说话人音色识别
        :param audio_np: 音频文件的numpy表示
        :return: 是否在参考聚类里
        """
        pass
