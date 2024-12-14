# anyasfriend/components/spk/campplus.py
from collections import Counter

import numpy as np
from loguru import logger

from anyasfriend.components.interfaces.spk import SpeakerRecognition

from .speakerlab import Diarization3Dspeaker


class CampplusSpeakerRecognition(SpeakerRecognition):
    def __init__(
        self,
        device: str = None,
        speaker_num: int = None,
        model_cache_dir: str = None,
        reference_nps: list[np.ndarray] = None,
    ):
        self.model = Diarization3Dspeaker(
            device=device, speaker_num=speaker_num, model_cache_dir=model_cache_dir
        )
        self.reference_labels_set = set()
        if reference_nps is not None:
            for ref_np in reference_nps:
                cluster_id = self.extract_features(ref_np)
                self.reference_labels_set.add(cluster_id)
        logger.info("Reference Voice(s): " + str(self.reference_labels_set))

    def extract_features(self, audio_np: np.ndarray) -> int:
        # Get the labels for the segments in the audio
        audio_labels = self.model(audio_np)

        # Collect all cluster_ids from the segments
        cluster_ids = [seg[2] for seg in audio_labels]
        print(cluster_ids)
        # Count the occurrences of each cluster_id and return the most frequent one
        counter = Counter(cluster_ids)
        most_common_cluster = counter.most_common(1)[0][0]

        return most_common_cluster

    def recognize(self, audio_np: np.ndarray) -> bool:
        # Get the most frequent cluster_id for the input audio
        recognized_cluster = self.extract_features(audio_np)
        print("recognized_cluster: ", recognized_cluster)
        # Check if the recognized cluster is in the set of reference clusters
        if recognized_cluster in self.reference_labels_set:
            return True
        return False


def main():
    """识别成功率有点低，暂时弃用了"""
    from anyasfriend.utils import load_audio_to_numpy

    ref_wav_list = [
        r"D:\PythonProject\原神语音\中文\胡桃\vo_BZLQ001_4_hutao_05.wav",
        r"D:\PythonProject\原神语音\中文\刻晴\vo_keqing_dialog_idle4.wav",
        r"D:\PythonProject\原神语音\中文\纳西妲\vo_LLZAQ001_4_nahida_07.wav",
    ]

    ref_nps = [load_audio_to_numpy(wav) for wav in ref_wav_list]

    campplus_model = CampplusSpeakerRecognition(reference_nps=ref_nps)
    test_wav_list = [
        r"D:\PythonProject\原神语音\中文\胡桃\vo_BZLQ001_4_hutao_06.wav",
        r"D:\PythonProject\原神语音\中文\刻晴\vo_keqing_dialog_idle3.wav",
        r"D:\PythonProject\原神语音\中文\派蒙\vo_ABDLQ002_1_paimon_02.wav",
        r"D:\PythonProject\原神语音\中文\纳西妲\vo_LLZAQ001_4_nahida_06.wav",
    ]

    audio_nps = [load_audio_to_numpy(wav) for wav in test_wav_list]

    for audio_np in audio_nps:
        res = campplus_model.recognize(audio_np)
        print(res)


if __name__ == "__main__":
    main()
