# anyasfriend/chatbot.py
import asyncio
from typing import Any, AsyncGenerator

from loguru import logger
from websockets.asyncio.client import ClientConnection

from anyasfriend.components.interfaces import ASR, LLM, TTS, VAD, Memory


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
        self.voice_output_queue = asyncio.Queue()
        self.current_task = None

        self.playback_assistant = self.tts.config.base.playback_assistant
        self.playback_user = self.tts.config.base.playback_user

    async def listen_for_text(self, websocket: ClientConnection):
        while True:
            text_input = await websocket.recv()  # 接收来自websocket的文字消息
            logger.info(f"[Text input]: {text_input}")
            await self.text_input_queue.put(text_input)  # 放入文字输入队列

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
                user_input = await self.text_input_queue.get()
                if self.current_task:
                    self.current_task.cancel()  # 如果当前有任务在执行，取消它
                self.current_task = asyncio.create_task(self.process_text(user_input))

            # 然后处理语音输入
            elif not self.voice_input_queue.empty():
                user_input = await self.voice_input_queue.get()
                if self.current_task:
                    self.current_task.cancel()  # 如果当前有任务在执行，取消它
                self.current_task = asyncio.create_task(self.process_voice(user_input))
            # 等待下一次输入
            await asyncio.sleep(0.1)

    async def process_text(self, input_text: str):
        response_stream = self.respond(input_text)
        audio_response = await self.speak(response_stream)
        return audio_response

    async def process_voice(self, input_text: str):
        response_stream = self.respond(input_text)
        audio_response = await self.speak(response_stream)
        return audio_response

    async def respond(self, input_text: str) -> AsyncGenerator[str, Any]:
        async for text in self.llm.generate_response(input_text):
            yield text

    async def speak(self, text_stream: AsyncGenerator[str, Any]) -> bytes:

        if False:
            with open("out.mp3", "wb") as f:
                async for chunk in self.tts.synthesize(response_text):
                    f.write(chunk)
                    audio_response += chunk

        audio_response = b""
        async for response_text in text_stream:
            logger.info(f"response_text: {response_text}")
            async for chunk in self.tts.synthesize(response_text):
                if self.playback_assistant:
                    await self.voice_output_queue.put(chunk)
                audio_response += chunk
        return audio_response

    async def chat(self):
        logger.info("聊天机器人准备好，等待对话...")
        await self.process_input()  # 只需要处理输入，监听工作已经通过 websockets.serve 做了
