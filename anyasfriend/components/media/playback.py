import asyncio
from concurrent.futures import ThreadPoolExecutor

from loguru import logger


# Audio playback function (run in a separate thread)
async def start_playback(audio_queue: asyncio.Queue, sr=16000):
    import pyaudio

    p = pyaudio.PyAudio()
    audio_format = pyaudio.paInt16
    stream = p.open(format=audio_format, channels=1, rate=sr, output=True)
    logger.info("Playback is running in a separate thread...")

    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=2)

    def remove_wav_header(audio_bytes):
        """Remove WAV header if it exists, and return only PCM data."""
        # WAV header is usually 44 bytes, starting with "RIFF"
        if (
            len(audio_bytes) > 44
            and audio_bytes[:4] == b"RIFF"
            and audio_bytes[8:12] == b"WAVE"
        ):
            # logger.debug("WAV header detected, removing header.")
            return audio_bytes[44:]  # Skip the 44-byte header
        return audio_bytes

    async def play_audio(audio_bytes):
        await loop.run_in_executor(executor, stream.write, audio_bytes)

    while True:
        audio_bytes = await audio_queue.get()
        if audio_bytes is None:
            logger.info("Received None, stop consuming...")
            break
        audio_bytes = remove_wav_header(audio_bytes)
        await play_audio(audio_bytes)
