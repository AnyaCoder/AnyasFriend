import asyncio
import sys
from typing import Any, AsyncGenerator

import websockets
from loguru import logger
from websockets.asyncio.client import ClientConnection

from anyasfriend.components.interfaces import ASR, LLM, TTS, VAD, Memory
from anyasfriend.components.media import Playback, PlaybackEvent

logger.remove()
logger.add(sys.stdout, level="DEBUG")


class Chatbot:
    def __init__(self, *, asr: ASR, llm: LLM, tts: TTS, vad: VAD, memory: Memory):
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.vad = vad
        self.memory = memory
        self.text_input_queue = asyncio.Queue()
        self.voice_input_queue = asyncio.Queue()
        self.current_task = None
        self.playback = Playback(
            sample_rate=tts.config.base.playback_sample_rate,
            frames_per_buffer=tts.config.base.playback_frames_per_buffer,
        )
        self.playback_assistant = self.tts.config.base.playback_assistant
        self.playback_user = self.tts.config.base.playback_user
        self.timeout = 2.0
        self.last_task_id = "Task-?"
        self.last_text = "<?>"

        self.text_clients: set[ClientConnection] = (
            set()
        )  # 用于存储所有的文本 WebSocket 客户端连接
        self.voice_clients: set[ClientConnection] = (
            set()
        )  # 用于存储所有的语音 WebSocket 客户端连接

        self.tts_text_queue = asyncio.Queue()
        self.subtitle_text_queue = asyncio.Queue()
        self.tts_audio_queue = asyncio.Queue()

    async def listen_for_text(self, websocket: ClientConnection, path: str = ""):
        logger.info(f"[Text input]: Client connected {websocket.remote_address}")
        self.text_clients.add(websocket)
        try:
            async for text in websocket:
                logger.info(f"[Text input]: {text}")
                await self.text_input_queue.put(text)
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"Text connection closed: {websocket.remote_address}")
        finally:
            self.text_clients.discard(websocket)

    async def listen_for_voice(self, websocket: ClientConnection, path: str = ""):
        logger.info(f"[Voice input]: Client connected {websocket.remote_address}")
        self.voice_clients.add(websocket)
        try:
            async for chunk in websocket:
                for audio_bytes in self.vad.detect_speech(chunk):
                    if self.playback.is_playing and audio_bytes == b"<|PAUSE|>":
                        self.playback.is_playing = False
                        if self.current_task:
                            self.current_task.cancel()
                            await self.playback.event_queue.put(PlaybackEvent.PAUSE)
                    elif not self.playback.is_playing and audio_bytes == b"<|RESUME|>":
                        self.playback.is_playing = True
                        await self.playback.event_queue.put(PlaybackEvent.RESUME)
                    elif self.playback.is_playing:
                        voice_input = await self.asr.recognize_speech(audio_bytes)
                        logger.info(f"[Voice input]: {voice_input}")
                        await self.voice_input_queue.put(voice_input)
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"Voice connection closed: {websocket.remote_address}")
        finally:
            self.voice_clients.discard(websocket)

    async def process_input(self):
        while True:
            if not self.text_input_queue.empty():
                user_input = await self.text_input_queue.get()
                await self.handle_input(user_input)

            elif not self.voice_input_queue.empty():
                user_input = await self.voice_input_queue.get()
                await self.handle_input(user_input, is_voice=True)

            await asyncio.sleep(0.01)

    async def handle_input(self, user_input: str, is_voice: bool = False):
        maybe_command = user_input.strip()

        if maybe_command.startswith("/"):
            await self.handle_command(maybe_command, is_voice)
            return

        await self.recreate_task(user_input, is_voice)

    async def handle_command(self, command: str, is_voice: bool):
        match command:
            case "/clear":
                self.memory.clear()
            case "/history":
                print("=" * 10 + " Chat History " + "=" * 10)
                print(self.memory.retrieve_all())
            case "/help":
                print("/clear /history /help /interrupt /pause /resume")
            case "/interrupt":
                await self.recreate_task(PlaybackEvent.PAUSE, is_voice=is_voice)
            case "/pause":
                await self.playback.event_queue.put(PlaybackEvent.PAUSE)
            case "/resume":
                await self.playback.event_queue.put(PlaybackEvent.RESUME)
            case _:
                print("Unknown command, see /help for more info.")

    async def recreate_task(self, user_input: str, is_voice: bool = False):
        if self.current_task:
            self.last_task_id = self.current_task.get_name()
            self.current_task.cancel()
            await self.playback.event_queue.put(PlaybackEvent.PAUSE)
            await self.playback.event_queue.put(PlaybackEvent.RESUME)
            while not self.tts_text_queue.empty():
                self.tts_text_queue.get_nowait()
            while not self.tts_audio_queue.empty():
                self.tts_audio_queue.get_nowait()
            while not self.subtitle_text_queue.empty():
                self.subtitle_text_queue.get_nowait()
        if user_input == PlaybackEvent.PAUSE:
            return
        task_func = self.process_voice if is_voice else self.process_text
        self.current_task = asyncio.create_task(task_func(user_input))

    async def process_text(self, input_text: str):
        await asyncio.gather(self.respond(input_text), self.text2audio(), self.speak())
        logger.info("[Text input]: Complete.")

    async def process_voice(self, input_text: str):
        await asyncio.gather(self.respond(input_text), self.text2audio(), self.speak())
        logger.info("[Voice output]: Complete.")

    async def respond(self, input_text: str):
        try:
            async for text in self.llm.generate_response(input_text):
                if (
                    self.last_task_id != self.current_task.get_name()
                    and self.last_text == text
                ):
                    logger.warning(
                        f"{self.last_task_id} != {self.current_task.get_name()}"
                    )
                    self.last_text = "<?>"
                    continue
                self.memory.store("assistant", text, delta=True)
                self.last_text = text
                await self.tts_text_queue.put(text)
                await self.subtitle_text_queue.put(text)
        except asyncio.CancelledError:
            logger.warning("Cancelled response")
            return
        logger.debug("**** respond **** Done")
        await self.tts_text_queue.put(None)
        await self.subtitle_text_queue.put("")  # clear

    async def text2audio(self):
        await self.tts_audio_queue.put(b"<|BOS|>")
        while True:
            text_chunk = await self.tts_text_queue.get()
            if text_chunk is None:
                break

            async for audio_chunk in self.tts.synthesize(text_chunk):
                await self.tts_audio_queue.put(audio_chunk)
            await self.tts_audio_queue.put(b"")  # end playback
            await self.tts_audio_queue.put(b"<|EOS|>")
        logger.debug("**** text2audio **** Done")
        await self.tts_audio_queue.put(None)

    async def speak(self):
        first_sentence = True

        async def send_text():
            # 取发送队列
            subtitle_text = await self.subtitle_text_queue.get()
            await self.send_text_response(subtitle_text)

        while True:
            audio_chunk = await self.tts_audio_queue.get()
            if audio_chunk is None:
                break

            elif audio_chunk == b"<|BOS|>":
                # print(audio_chunk, end=' ')
                if first_sentence:
                    first_sentence = False
                    await send_text()

            elif audio_chunk == b"<|EOS|>":
                # 等这一句播放完
                # print(audio_chunk)
                await self.playback.play_complete.wait()
                if not first_sentence:
                    await send_text()
                self.playback.play_complete.clear()

            elif self.playback_assistant and self.playback.is_playing:
                # print(f"<audio: {len(audio_chunk)} bytes>", end=' ')
                self.playback.play_audio(audio_chunk)
                await self.send_audio_response(audio_chunk)

        logger.debug("**** speak **** Done")

    async def send_text_response(self, text: str):
        for websocket in self.text_clients:
            try:
                await asyncio.wait_for(websocket.send(text), timeout=self.timeout)
            except asyncio.exceptions.TimeoutError:
                logger.warning(f"[Text send] Timeout for {self.timeout} secs, ignored")
            except websockets.exceptions.ConnectionClosed:
                self.text_clients.discard(websocket)

    async def send_audio_response(self, audio_data: bytes):
        for websocket in self.voice_clients:
            try:
                await asyncio.wait_for(websocket.send(audio_data), timeout=self.timeout)
            except asyncio.exceptions.TimeoutError:
                logger.warning(f"[Audio send] Timeout for {self.timeout} secs, ignored")
            except websockets.exceptions.ConnectionClosed:
                self.voice_clients.discard(websocket)

    async def chat(
        self,
        text_ws_host: str,
        text_ws_port: int,
        voice_ws_host: str,
        voice_ws_port: int,
    ):
        logger.info("初始化...完成。")
        logger.info(f"文本端: ws://{text_ws_host}:{text_ws_port}")
        logger.info(f"语音端: ws://{voice_ws_host}:{voice_ws_port}")
        self.text_websocket = websockets.serve(
            self.listen_for_text,
            text_ws_host,
            text_ws_port,
        )
        self.voice_websocket = websockets.serve(
            self.listen_for_voice,
            voice_ws_host,
            voice_ws_port,
        )
        await asyncio.gather(
            self.text_websocket,
            self.voice_websocket,
            self.process_input(),
            self.playback.start_playback(),
        )
