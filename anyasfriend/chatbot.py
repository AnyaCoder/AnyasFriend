import asyncio
import sys
import uuid
from datetime import datetime
from uuid import UUID

import ormsgpack
import websockets
from loguru import logger
from tqdm.asyncio import tqdm
from websockets.asyncio.client import ClientConnection

from anyasfriend.components.interfaces import ASR, LLM, TTS, VAD, Core, Memory
from anyasfriend.schema import AnyaData

logger.remove()
logger.add(sys.stdout, level="INFO")


class Chatbot(Core):
    def __init__(self, *, asr: ASR, llm: LLM, tts: TTS, vad: VAD, memory: Memory):
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.vad = vad
        self.memory = memory
        self.text_input_queue = asyncio.Queue()
        self.voice_input_queue = asyncio.Queue()
        self.llm_prompt_queue = asyncio.Queue()

        self.timeout = 2.0

        self.clients: set[ClientConnection] = (
            set()
        )  # 用于存储所有的 WebSocket 客户端连接

        self.tts_text_queue = asyncio.Queue()
        self.temp_text_queue = asyncio.Queue()
        self.tts_audio_queue = asyncio.Queue()
        self.accepted_identifier = uuid.uuid4()
        self.continue_event.set()

        self.only_one_text_event = asyncio.Event()
        self.only_one_audio_event = asyncio.Event()
        self.text_first_event = asyncio.Event()

    async def _data_wrapper(self, websocket):
        async for data in websocket:
            yield data

    async def send_back_text(self):
        if self.only_one_text_event.is_set():
            return  # 不能同时存在两个ACCEPT_TEXT, 得到一个之前必须把这个消耗掉
        self.only_one_text_event.set()
        # logger.debug("recv ACCEPT_TEXT")
        unique_id, text = await self.tts_text_queue.get()
        await self.send_response(AnyaData.Type.TEXT, text, unique_id)
        # logger.debug(f"send_back_text: {text}")
        self.only_one_text_event.clear()
        self.text_first_event.set()  # 文本发完了

    async def send_back_audio(self):
        if self.only_one_audio_event.is_set():
            return  # 不能同时存在两个ACCEPT_AUDIO, 得到一个之前必须把这个消耗掉
        self.only_one_audio_event.set()
        await self.text_first_event.wait()  # 得等文本发完
        # logger.debug("recv ACCEPT_AUDIO")
        unique_id, audio = await self.tts_audio_queue.get()
        await self.send_response(AnyaData.Type.AUDIO, audio, unique_id)
        # logger.debug("send_back_audio")
        self.only_one_audio_event.clear()

    async def handle_client(self, websocket: ClientConnection, path: str = ""):
        logger.info(f"[Client connected]: {websocket.remote_address}")
        self.clients.add(websocket)
        try:
            async for raw_data in tqdm(
                self._data_wrapper(websocket), desc="Incoming Data"
            ):
                data_obj: dict = ormsgpack.unpackb(raw_data)
                data = AnyaData(**data_obj)
                # 输入文本
                if data.dtype == AnyaData.Type.TEXT:
                    text = data.content.strip()
                    if text:
                        logger.info(f"[Text input]: {text}")
                        await self.text_input_queue.put(
                            (self.accepted_identifier, text)
                        )
                # 输入声音
                elif data.dtype == AnyaData.Type.AUDIO:
                    chunk = data.content

                    for audio_bytes in self.vad.detect_speech(chunk):
                        if audio_bytes == b"<|PAUSE|>":
                            self.cancel_event.set()
                        elif audio_bytes == b"<|RESUME|>":
                            pass
                        elif len(audio_bytes) > 1024:
                            voice_input = await self.asr.recognize_speech(audio_bytes)
                            logger.info(f"[Voice input]: {voice_input}")
                            await self.voice_input_queue.put(
                                (self.accepted_identifier, voice_input)
                            )
                # 触发事件
                elif data.dtype == AnyaData.Type.EVENT:
                    event = data.content
                    match event:
                        case AnyaData.Event.ACCEPT_TEXT.value:
                            if self.accepted_identifier != data.identifier:
                                # 不相等说明要么是打断，要么是新文本，得保证传回合成文本先于传回合成的音频
                                self.accepted_identifier = (
                                    data.identifier
                                )  # 设置允许的uuid
                                self.text_first_event.clear()  # 取消事件标志，让 send_back_audio 等待
                                logger.debug(
                                    "UPDATED accepted_identifier: "
                                    + str(self.accepted_identifier)
                                )
                            asyncio.create_task(self.send_back_text())
                        case AnyaData.Event.ACCEPT_AUDIO.value:
                            asyncio.create_task(self.send_back_audio())
                        case AnyaData.Event.CANCEL.value:
                            self.cancel_event.set()  # 取消事件
                        case _:
                            logger.warning("No such AnyaData.Event")
                # handle others
                else:
                    raise NotImplementedError
        except Exception as e:
            logger.warning(
                f"Client connection closed: {websocket.remote_address},\
                caused by: {e}"
            )
        finally:
            self.clients.discard(websocket)

    async def process_input(self):
        while True:
            if not self.text_input_queue.empty():
                unique_id, user_input = await self.text_input_queue.get()
                await self.handle_input(unique_id, user_input)
                logger.info("[Text input]: Complete.")

            elif not self.voice_input_queue.empty():
                unique_id, user_input = await self.voice_input_queue.get()
                await self.handle_input(unique_id, user_input, is_voice=True)
                logger.info("[Voice input]: Complete.")

            await asyncio.sleep(0.1)

    async def handle_input(
        self, unique_id: UUID, user_input: str, is_voice: bool = False
    ):
        maybe_command = user_input.strip()

        if maybe_command.startswith("/"):
            await self.handle_command(maybe_command, is_voice)
            return

        prompt = maybe_command
        await self.llm_prompt_queue.put((unique_id, prompt))

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
                print("interrupt")
            case _:
                print("Unknown command, see /help for more info.")

    async def clean_queued_data(self):
        logger.warning("clean_queued_data")
        while not self.tts_text_queue.empty():
            await self.tts_text_queue.get()
        while not self.temp_text_queue.empty():
            await self.temp_text_queue.get()
        while not self.tts_audio_queue.empty():
            await self.tts_audio_queue.get()

    async def wait_cancel_task(self):
        while True:
            await self.cancel_event.wait()
            self.continue_event.clear()  # 不允许继续
            # 尝试优雅退出，不用cancel
            await asyncio.sleep(0.2)  # 交还控制权，等该取消的任务退出来
            await self.clean_queued_data()  # 异步清理
            await self.send_response(AnyaData.Type.TEXT, "", self.accepted_identifier)
            await self.send_response(AnyaData.Type.AUDIO, b"", self.accepted_identifier)
            self.cancel_event.clear()  # 恢复等下一次取消
            self.continue_event.set()  # 允许继续

    async def respond_with_text(self):
        while True:
            unique_id, prompt = await self.llm_prompt_queue.get()
            print(unique_id, prompt)
            await self.continue_event.wait()  # 能否继续？

            # 创建一个队列来存储生成的文本
            done_event = asyncio.Event()

            # 定义生成文本的任务
            async def generate_text():
                try:
                    async for text in self.llm.generate_response(prompt):
                        if self.cancel_event.is_set():
                            break
                        # 获取生成的文本
                        self.memory.store("assistant", text, delta=True)
                        await self.tts_text_queue.put((unique_id, text))
                        await self.temp_text_queue.put((unique_id, text))

                except asyncio.CancelledError:
                    logger.warning("Text generation cancelled.")
                finally:
                    done_event.set()
                    await self.tts_text_queue.put((unique_id, ""))

            # 并行执行生成文本和检查取消事件
            gen_task = asyncio.create_task(generate_text())

            # 定义检查取消事件的任务
            async def check_cancel_event():
                while not self.cancel_event.is_set() and not done_event.is_set():
                    await asyncio.sleep(0.05)  # 定期检查取消事件
                if self.cancel_event.is_set():
                    gen_task.cancel()

            check_task = asyncio.create_task(check_cancel_event())

            # 等待生成任务和取消检查任务完成
            await asyncio.gather(gen_task, check_task)
        pass

    async def respond_with_audio(self):
        while True:
            await self.continue_event.wait()  # 检查是否能继续

            unique_id, text_chunk = await self.temp_text_queue.get()
            if text_chunk is None:
                continue

            done_event = asyncio.Event()  # 创建一个事件，用于标记音频生成的完成

            # 定义生成音频的任务
            async def generate_audio():
                try:
                    async for audio_chunk in self.tts.synthesize(text_chunk):
                        if self.cancel_event.is_set():
                            break
                        await self.tts_audio_queue.put((unique_id, audio_chunk))
                except asyncio.CancelledError:
                    logger.warning("Audio generation cancelled.")
                finally:
                    done_event.set()  # 确保任务完成时标记
                    await self.tts_audio_queue.put(
                        (unique_id, b"")
                    )  # 确保音频停止后输出空音频

            gen_task = asyncio.create_task(generate_audio())

            # 定义检查取消事件的任务
            async def check_cancel_event():
                while not self.cancel_event.is_set() and not done_event.is_set():
                    await asyncio.sleep(0.05)  # 定期检查取消事件
                if self.cancel_event.is_set():
                    gen_task.cancel()

            # 并行执行生成音频和检查取消事件
            check_task = asyncio.create_task(check_cancel_event())

            # 等待生成任务和取消检查任务完成
            await asyncio.gather(gen_task, check_task)
        pass

    async def send_response(
        self, dtype: AnyaData.Type, raw_data: str | bytes, identifier: UUID
    ):
        data = AnyaData(
            dtype=dtype,
            content=raw_data,
            identifier=identifier,
            timestamp=datetime.now(),
        )
        packed_data = ormsgpack.packb(data, option=ormsgpack.OPT_SERIALIZE_PYDANTIC)
        for websocket in self.clients:
            try:
                await asyncio.wait_for(
                    websocket.send(packed_data), timeout=self.timeout
                )
            except asyncio.exceptions.TimeoutError:
                logger.warning(
                    f"[Server send to Client] Timeout for {self.timeout} secs, ignored"
                )
            except Exception as e:
                logger.warning(f"[Client] lost connection due to: {e}")

    async def chat(
        self,
        server_ws_host: str,
        server_ws_port: int,
    ):
        logger.info("初始化...完成。")
        logger.info(f"服务端: ws://{server_ws_host}:{server_ws_port}")

        self.server_websocket = websockets.serve(
            self.handle_client,
            server_ws_host,
            server_ws_port,
        )
        await asyncio.gather(
            self.server_websocket,
            self.process_input(),
            self.wait_cancel_task(),
            self.respond_with_text(),
            self.respond_with_audio(),
        )
