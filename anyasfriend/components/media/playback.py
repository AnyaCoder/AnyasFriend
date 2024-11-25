import asyncio
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

import pyaudio
from loguru import logger


class PlaybackEvent(Enum):
    PAUSE = "<|pause|>"
    RESUME = "<|resume|>"
    STOP = "<|stop|>"


class Playback:
    def __init__(self, sr=16000):
        self.audio_queue = asyncio.Queue()
        self.event_queue = asyncio.Queue()
        self.sr = sr
        self.p = pyaudio.PyAudio()
        self.audio_format = pyaudio.paInt16
        self.stream = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.loop = asyncio.get_event_loop()
        self.is_playing = False  # state
        self.lock = asyncio.Lock()  # Ensure thread-safe access to stream

    def start_stream(self):
        self.is_playing = True
        self.stream = self.p.open(
            format=self.audio_format, channels=1, rate=self.sr, output=True
        )
        logger.info("Playback stream started.")

    async def stop_stream(self):
        async with self.lock:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                logger.info("Playback stream stopped.")
            self.p.terminate()
            logger.info("Audio playback terminated.")

    async def pause(self):
        async with self.lock:
            self.is_playing = False
            self.stream.stop_stream()
            self.clear_queue()
            logger.info("Playback paused and queue cleared.")

    async def resume(self):
        async with self.lock:
            self.is_playing = True
            self.stream.start_stream()
            logger.info("Playback resumed.")

    def remove_wav_header(self, audio_bytes: bytes) -> bytes:
        """Remove WAV header if it exists, and return only PCM data."""
        while (
            len(audio_bytes) > 44
            and audio_bytes[:4] == b"RIFF"
            and audio_bytes[8:12] == b"WAVE"
        ):
            audio_bytes = audio_bytes[44:]  # Skip the 44-byte header
        return audio_bytes

    async def play_audio(self, audio_bytes: bytes):
        """Play the audio data asynchronously using an executor."""
        async with self.lock:
            if self.is_playing and self.stream and self.stream.is_active():
                await self.loop.run_in_executor(
                    self.executor, self.stream.write, audio_bytes
                )
            else:
                # logger.warning("Attempted to write to a stopped or inactive stream.")
                pass

    async def handle_event(self, event: PlaybackEvent):
        logger.debug(f"Received event: {event}")
        if event == PlaybackEvent.PAUSE:
            await self.pause()
        elif event == PlaybackEvent.STOP:
            await self.stop_stream()
        elif event == PlaybackEvent.RESUME:
            await self.resume()

    async def loop_event(self):
        while True:
            event: PlaybackEvent = await self.event_queue.get()
            await self.handle_event(event)
            if event == PlaybackEvent.STOP:
                await self.audio_queue.put(None)
                break

    async def loop_data(self):
        while True:
            audio_bytes = await self.audio_queue.get()
            if audio_bytes is None:
                logger.info("Received None, stop consuming...")
                break

            audio_bytes = self.remove_wav_header(audio_bytes)
            await self.play_audio(audio_bytes)

    async def start_playback(self):
        logger.info("Playback is running in a separate thread...")
        self.start_stream()

        await asyncio.gather(self.loop_data(), self.loop_event())

        # Cleanup and stop playback
        await self.stop_stream()

    def clear_queue(self):
        while not self.audio_queue.empty():
            self.audio_queue.get_nowait()


async def main():
    playback = Playback(sr=44100)

    # Start the playback task
    playback_task = asyncio.create_task(playback.start_playback())

    async def data_events(file_path: str, chunk_size: int = 16384):
        logger.info("Data coming...")
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                await asyncio.sleep(0.01)
                await playback.audio_queue.put(chunk)

    # Control events, can be done concurrently with playback
    async def control_events():
        await asyncio.sleep(3)
        logger.info("Pausing playback...")
        await playback.event_queue.put(PlaybackEvent.PAUSE)
        await asyncio.sleep(2)
        logger.info("Resuming playback...")
        await playback.event_queue.put(PlaybackEvent.RESUME)
        await asyncio.sleep(1)
        logger.info("Stopping playback...")
        await playback.event_queue.put(PlaybackEvent.STOP)

    file_path = r"D:\PythonProject\原神语音\中文\胡桃\vo_EQHDJ201_1_talk_hutao_14.wav"

    await asyncio.gather(playback_task, data_events(file_path), control_events())


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
