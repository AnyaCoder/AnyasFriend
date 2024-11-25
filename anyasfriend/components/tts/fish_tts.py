# anyasfriend/components/tts/fish_tts.py

from typing import Literal

import ormsgpack
from loguru import logger
from pydantic import BaseModel

from anyasfriend.components.interfaces import TTS, AnyTTSConfig
from anyasfriend.config import config as global_config


class ServeReferenceAudio(BaseModel):
    audio: bytes
    text: str


class FishTTSRequestConfig(BaseModel):
    chunk_length: int = 200
    # Audio format
    format: Literal["wav", "pcm", "mp3"] = "wav"
    mp3_bitrate: Literal[64, 128, 192] = 128
    # References audios for in-context learning
    references: list[ServeReferenceAudio] = []
    # Reference id
    # For example, if you want use https://fish.audio/m/7f92f8afb8ec43bf81429cc1c9199cb1/
    # Just pass 7f92f8afb8ec43bf81429cc1c9199cb1
    reference_id: str | None = None
    # Normalize text for en & zh, this increase stability for numbers
    normalize: bool = True
    # Balance mode will reduce latency to 300ms, but may decrease stability
    latency: Literal["normal", "balanced"] = "balanced"
    # not usually used below
    streaming: bool = True
    top_p: float = 0.7
    repetition_penalty: float = 1.2
    temperature: float = 0.7


class FishTTSConfig(AnyTTSConfig):
    request: FishTTSRequestConfig = FishTTSRequestConfig()  # default


class FishTTSRequest(FishTTSRequestConfig):
    text: str


class FishTTS(TTS):

    def __init__(
        self,
        config: FishTTSConfig,
    ):
        super().__init__(config)
        self.frames_per_buffer = 16384

        ref_audios, ref_texts = [], []
        for audio_file in global_config.chatbot.tts.reference_audios:
            with open(audio_file, "rb") as f:
                ref_audios.append(f.read())

        for text_file in global_config.chatbot.tts.reference_texts:
            with open(text_file, "r", encoding="utf-8") as f:
                ref_texts.append(f.read())

        assert len(ref_texts) == len(
            ref_audios
        ), "The number of reference audios and texts do not match."

        self.config.request.references = [
            ServeReferenceAudio(audio=a, text=t) for a, t in zip(ref_audios, ref_texts)
        ]
        logger.info(f"FishTTS initalized!")

    async def synthesize(self, text: str):
        tts_request = FishTTSRequest(**self.config.request.model_dump(), text=text)
        # logger.info(f"tts: {text}")
        async with self.client.stream(
            method="POST",
            url=self.config.base.api_url,
            data=ormsgpack.packb(
                tts_request,
                option=ormsgpack.OPT_SERIALIZE_PYDANTIC,
            ),
            headers={
                "authorization": f"Bearer {self.config.base.api_key}",
                "Content-Type": "application/msgpack",
            },
        ) as response:
            if response.status_code != 200:
                raise ValueError(
                    f"Failed to get response, status code: {response.status_code}"
                )
            async for chunk in response.aiter_bytes(chunk_size=self.frames_per_buffer):
                yield chunk

    async def adjust_params(self, params: FishTTSConfig) -> None:
        self.config = params


async def tts_main():
    from anyasfriend.config import config

    config_dict = dict(
        base=dict(
            api_key=config.chatbot.tts.api_key,
            api_url=config.chatbot.tts.api_url,
        ),
        # request=dict(
        #     format="mp3",
        # )
    )
    tts = FishTTS(config=FishTTSConfig(**config_dict))
    text = "你好，我真的很喜欢你!"
    audio = f"output.{tts.config.request.format}"
    with open(audio, "wb") as f:
        async for chunk in tts.synthesize(text):
            f.write(chunk)


if __name__ == "__main__":
    import asyncio

    asyncio.run(tts_main())
