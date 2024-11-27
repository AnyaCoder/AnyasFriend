# anyasfriend/chatbot.py
import asyncio
import sys
from typing import Any, AsyncGenerator

from loguru import logger
from websockets.asyncio.client import ClientConnection

from anyasfriend.components.interfaces import ASR, LLM, TTS, VAD, Memory
from anyasfriend.components.media import Playback, PlaybackEvent

logger.remove()
logger.add(sys.stdout, level="INFO")


class Chatbot:
    def __init__(
        self,
        *,
        asr: ASR,
        llm: LLM,
        tts: TTS,
        vad: VAD,
        memory: Memory,
    ):
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.vad = vad
        self.memory = memory
        self.text_input_queue = asyncio.Queue()
        self.voice_input_queue = asyncio.Queue()
        self.current_task = None
        self.playback = Playback(sr=44100)
        self.playback_assistant = self.tts.config.base.playback_assistant
        self.playback_user = self.tts.config.base.playback_user

    async def listen_for_text(self, websocket: ClientConnection):
        async for text in websocket:
            logger.info(f"[Text input]: {text}")
            await self.text_input_queue.put(text)  # 放入文字输入队列

    async def listen_for_voice(self, websocket: ClientConnection):
        async for chunk in websocket:
            for audio_bytes in self.vad.detect_speech(chunk):
                voice_input = await self.asr.recognize_speech(audio_bytes)
                logger.info(f"[Voice input]: {voice_input}")
                await self.voice_input_queue.put(voice_input)  # 放入语音输入队列

    async def process_input(self):
        while True:
            # 优先处理文字输入
            if not self.text_input_queue.empty():
                user_input: str = await self.text_input_queue.get()
                await self.handle_input(user_input)

            # 处理语音输入
            elif not self.voice_input_queue.empty():
                user_input = await self.voice_input_queue.get()
                await self.handle_input(user_input, is_voice=True)

            # 等待下一次输入
            await asyncio.sleep(0.01)

    async def handle_input(self, user_input: str, is_voice: bool = False):
        maybe_command = user_input.strip()

        # 处理命令
        if maybe_command.startswith("/"):
            await self.handle_command(maybe_command, is_voice)
            return  # 处理完命令后返回，不再继续处理输入

        # 处理普通输入
        await self.recreate_task(user_input, is_voice)

    async def handle_command(self, command: str, is_voice: bool):
        """处理命令输入"""
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
        """处理普通输入, 取消当前任务并启动新的任务"""
        if self.current_task:
            self.current_task.cancel()
            await self.playback.event_queue.put(PlaybackEvent.PAUSE)
            await self.playback.event_queue.put(PlaybackEvent.RESUME)
        if user_input == PlaybackEvent.PAUSE:
            return
        task_func = self.process_voice if is_voice else self.process_text
        self.current_task = asyncio.create_task(task_func(user_input))

    async def process_text(self, input_text: str):
        response_stream = self.respond(input_text)
        audio_response = await self.speak(response_stream)
        logger.info("[Text input]: Complete.")
        return audio_response

    async def process_voice(self, input_text: str):
        response_stream = self.respond(input_text)
        audio_response = await self.speak(response_stream)
        logger.info("[Voice output]: Complete.")
        return audio_response

    async def respond(self, input_text: str) -> AsyncGenerator[str, Any]:
        async for text in self.llm.generate_response(input_text):
            self.memory.store("assistant", text, delta=True)
            yield text

    async def speak(self, text_stream: AsyncGenerator[str, Any]) -> bytes:

        audio_response = bytearray()
        try:
            async for text_chunk in text_stream:
                async for audio_chunk in self.tts.synthesize(text_chunk):
                    if self.playback_assistant:
                        await self.playback.audio_queue.put(audio_chunk)
                    audio_response.extend(audio_chunk)
        except asyncio.CancelledError:
            logger.warning("Speak/Text task was cancelled.")
            raise

        return bytes(audio_response)

    async def chat(self):
        logger.info("聊天机器人准备好，等待对话...")
        # 只需要处理输入，监听工作已经通过 websockets.serve 做了
        # 同时把语音打开
        await asyncio.gather(self.process_input(), self.playback.start_playback())
