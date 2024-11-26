# anyasfriend/components/asr/fun_asr.py
import os

os.environ["MODELSCOPE_CACHE"] = os.path.join(os.environ.get("TEMP", "."), "funasr")
import asyncio
import re
import time
from typing import List, Literal

import numpy as np
import torch
import torchaudio
from funasr import AutoModel
from funasr.download.download_model_from_hub import name_maps_ms
from loguru import logger
from pydantic import BaseModel

from anyasfriend.components.interfaces import ASR

global_lock = asyncio.Lock()


PROMPT = {
    "zh": "人间灯火倒映湖中，她的渴望让静水泛起涟漪。若代价只是孤独，那就让这份愿望肆意流淌。",
    "en": "In the realm of advanced technology, the evolution of artificial intelligence stands as a monumental achievement.",
    "ja": "先進技術の領域において、人工知能の進化は画期的な成果として立っています。常に機械ができることの限界を押し広げているこのダイナミックな分野は、急速な成長と革新を見せています。複雑なデータパターンの解読から自動運転車の操縦まで、AIの応用は広範囲に及びます。",
}


class ServeASRTranscription(BaseModel):
    text: str
    duration: float
    huge_gap: bool


class FunASRConfig(BaseModel):
    model: str = "iic/SenseVoiceSmall"
    device: str = "cuda"
    hub: str = "ms"
    # vad_model: str = "fsmn-vad"
    punc_model: str = "ct-punc"
    log_level: str = "ERROR"
    disable_pbar: bool = True
    disable_update: bool = False

    sample_rate: int = 16000
    language: Literal["zh", "en", "jp", "auto"] = "auto"


class FunASR(ASR):
    def __init__(self, config: FunASRConfig):
        if config.disable_update:
            config.model = os.path.join(
                os.getenv("MODELSCOPE_CACHE"), "hub", config.model
            )
            config.punc_model = os.path.join(
                os.getenv("MODELSCOPE_CACHE"), "hub", name_maps_ms[config.punc_model]
            )
            logger.warning(
                f"Disabled auto update, use local: \nstt:{config.model}\npunc:{config.punc_model}"
            )
        self.config = config
        self.model = self.load_asr_model()
        logger.info("FunASR initalized")

    def load_asr_model(self):
        return AutoModel(**self.config.model_dump())

    async def recognize_speech(self, audio_data: bytes) -> str:
        start_time = time.time()
        origin_audios = [audio_data]
        audios = [
            (np.frombuffer(audio, dtype=np.int16).astype(np.float32).copy() / 32767)
            for audio in origin_audios
        ]
        audios = [torch.from_numpy(audio).float() for audio in audios]

        if any(audios.shape[-1] >= 30 * self.config.sample_rate for audios in audios):
            raise Exception(status_code=400, detail="Audio length is too long")

        transcriptions: List[ServeASRTranscription] = await self.batch_asr(
            self.model,
            audios=audios,
            sr=self.config.sample_rate,
            language=self.config.language,
        )

        logger.info(f"[EXEC] ASR time: {(time.time() - start_time) * 1000:.2f}ms")
        return transcriptions[0].text

    async def batch_asr(self, model, audios, sr, language="auto"):
        resampled_audios = await asyncio.gather(
            *[self.resample_audio(audio, sr) for audio in audios]
        )

        async with global_lock:
            if language in PROMPT.keys():
                res = await self.async_generate(model, resampled_audios, language)
            else:
                res = await self.async_generate(model, resampled_audios)

        return await asyncio.gather(
            *[self.process_transcription(r, audio, sr) for r, audio in zip(res, audios)]
        )

    async def resample_audio(self, audio, sr):
        return await asyncio.to_thread(torchaudio.functional.resample, audio, sr, 16000)

    async def async_generate(self, model, resampled_audios, language=None):
        return await asyncio.to_thread(
            model.generate,
            input=resampled_audios,
            batch_size=300,
            hotword=PROMPT.get(language),
            merge_vad=True,
            merge_length_s=15,
            use_itn=True,
        )

    async def process_transcription(self, r, audio, sr):
        text = r["text"]
        text = re.sub(r"<\|.*?\|>", "", text).strip()
        duration = len(audio) / sr * 1000
        huge_gap = False

        if "timestamp" in r and len(r["timestamp"]) > 2:
            for timestamp_a, timestamp_b in zip(
                r["timestamp"][:-1], r["timestamp"][1:]
            ):
                if timestamp_b[0] - timestamp_a[1] > 5000:
                    huge_gap = True
                    break

            if duration - r["timestamp"][-1][1] > 3000:
                huge_gap = True

        return ServeASRTranscription(text=text, duration=duration, huge_gap=huge_gap)


async def asr_main():
    audio_file = r"D:\PythonProject\原神语音\中文\胡桃\vo_BZLQ001_4_hutao_13.wav"
    with open(audio_file, "rb") as f:
        b = f.read()
    asr = FunASR(config=FunASRConfig(sample_rate=44100))
    res = await asr.recognize_speech(b)
    logger.info(f"res: {res}")


if __name__ == "__main__":
    asyncio.run(asr_main())
