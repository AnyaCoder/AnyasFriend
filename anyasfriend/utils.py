import io
import os

import librosa


def load_audio_to_numpy(audio_input, sr: int = None):
    """
    加载音频文件并转换为一维numpy数组，支持文件路径和字节数据。

    Parameters:
    - audio_input: 音频文件的路径或音频字节数据
    - sr: 采样率 (默认16kHz) (None为保持原有采样率)

    Returns:
    - 一维 numpy 数组，表示音频信号
    """
    if isinstance(audio_input, str) and os.path.isfile(audio_input):
        # 如果是文件路径
        y, sr_actual = librosa.load(audio_input, sr=None, mono=True)
    elif isinstance(audio_input, bytes):
        # 如果是字节数据
        audio_bytes = io.BytesIO(audio_input)
        y, sr_actual = librosa.load(audio_bytes, sr=None, mono=True)
    else:
        raise ValueError(
            "audio_input must be either a valid file path or audio byte data."
        )

    # 如果采样率不匹配，则进行重采样
    if sr is not None and sr != sr_actual:
        y = librosa.resample(y, sr_actual, sr)

    return y
