# anyasfriend/components/tts/edge_tts.py

import asyncio
from io import BytesIO

import edge_tts
import soundfile as sf
from loguru import logger
from pydantic import BaseModel
from pydub import AudioSegment

from anyasfriend.components.interfaces import TTS, AnyTTSConfig


class EdgeTTSRequestConfig(BaseModel):
    # en-US-AvaMultilingualNeural
    # en-US-EmmaMultilingualNeural
    # en-US-JennyNeural
    voice: str = "en-US-AvaMultilingualNeural"


class EdgeTTSConfig(AnyTTSConfig):
    request: EdgeTTSRequestConfig = EdgeTTSRequestConfig()  # default


class EdgeTTSRequest(EdgeTTSRequestConfig):
    text: str


class EdgeTTS(TTS):
    def __init__(self, config: EdgeTTSConfig):
        super().__init__(config)

        logger.info(f"EdgeTTS initalized!")

    async def synthesize(self, text: str):
        communicate = edge_tts.Communicate(text, self.config.request.voice)
        mp3_bytes = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_bytes.extend(chunk["data"])
        audio_bytes = mp3_to_wav(bytes(mp3_bytes))
        i = 0
        while i < len(audio_bytes):
            yield audio_bytes[i : i + 16384]
            i += 16384

    async def adjust_params(self, params: EdgeTTSConfig) -> None:
        self.config = params


def mp3_to_wav(mp3_bytes: bytes) -> bytes:
    audio = AudioSegment.from_mp3(BytesIO(mp3_bytes))
    wav_output = BytesIO()
    audio.export(wav_output, format="wav")
    wav_output.seek(0)
    wav_data, samplerate = sf.read(wav_output)
    wav_byte_stream = BytesIO()
    sf.write(wav_byte_stream, wav_data, samplerate, format="WAV")
    wav_byte_stream.seek(0)
    wav_bytes = wav_byte_stream.read()

    return wav_bytes


async def edge_tts_main():
    from anyasfriend.config import config

    config_dict = dict(
        base=dict(
            api_key=config.chatbot.tts.api_key,
            api_url=config.chatbot.tts.base_url,
        ),
        # request=dict(
        #     format="mp3",
        # )
    )
    tts = EdgeTTS(config=EdgeTTSConfig(**config_dict))
    audio = "out.wav"
    text = "你好，我真的不喜欢你。"
    with open(audio, "wb") as f:
        async for chunk in tts.synthesize(text):
            f.write(chunk)


if __name__ == "__main__":
    import asyncio

    asyncio.run(edge_tts_main())
