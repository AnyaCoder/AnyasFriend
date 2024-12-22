import asyncio
import re
import sys
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Set
from uuid import UUID

import ormsgpack
import websockets
from loguru import logger
from websockets.asyncio.client import ClientConnection

from anyasfriend.components.interfaces import ASR, LLM, TTS, VAD, Memory
from anyasfriend.schema import AnyaData

# Configure logger
logger.remove()
logger.add(sys.stdout, level="INFO")


class ClientSession:
    """
    Represents a single client's session, maintaining its own queues, events, and memory.
    """

    def __init__(
        self,
        websocket: ClientConnection,
        asr: ASR,
        llm: LLM,
        tts: TTS,
        vad: VAD,
        memory_cls: Memory,
    ):
        self.websocket = websocket
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.vad = vad
        self.memory: Memory = memory_cls()  # 每个会话有独立的 Memory 实例

        # 独立的队列
        self.text_input_queue: asyncio.Queue = asyncio.Queue()
        self.voice_input_queue: asyncio.Queue = asyncio.Queue()
        self.llm_prompt_queue: asyncio.Queue = asyncio.Queue()
        self.llm_text_queue: asyncio.Queue = asyncio.Queue()
        self.tool_call_queue: asyncio.Queue = asyncio.Queue()
        self.tts_text_queue: asyncio.Queue = asyncio.Queue()
        self.tts_audio_queue: asyncio.Queue = asyncio.Queue()

        # 独立的事件
        self.cancel_event = asyncio.Event()
        self.continue_event = asyncio.Event()
        self.continue_event.set()

        self.only_one_text_event = asyncio.Event()
        self.only_one_audio_event = asyncio.Event()
        self.text_first_event = asyncio.Event()

        # 会话唯一标识符
        self.accepted_identifier: UUID = uuid.uuid4()

        # Timeout 配置
        self.timeout: float = 2.0

        # Function calling 配置
        self.func_calling: bool = self.llm.config.base.func_calling

        # 预编译正则表达式
        self.bracket_patterns = re.compile(r"[\(\[（【][^()\[\]（）【】]*[\)\]）】]")

        # 会话历史
        self.chat_history_events: asyncio.Event = asyncio.Event()
        self.chat_history_messages: List[Dict[str, str]] = []

        # Unique identifier for the current session
        self.identifier: UUID = uuid.uuid4()


class Chatbot:
    """
    Chatbot class that handles multiple WebSocket connections, each with its own session.
    """

    def __init__(self, *, asr: ASR, llm: LLM, tts: TTS, vad: VAD, memory_cls: Memory):
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.vad = vad
        self.memory_cls = memory_cls  # 注意这里传入的是 Memory 类，而不是实例

        # 集合来维护所有活跃的客户端会话
        self.sessions: Set[ClientSession] = set()

    async def _data_wrapper(self, websocket: ClientConnection):
        """Asynchronous generator to yield data from the websocket."""
        async for data in websocket:
            yield data

    async def send_response(
        self,
        dtype: AnyaData.Type,
        raw_data: str | bytes | dict | UUID,
        identifier: UUID,
        session: ClientSession,
    ):
        """
        Send a response to a specific client.

        Args:
            dtype (AnyaData.Type): The type of data (TEXT, AUDIO, EVENT).
            raw_data (str | bytes | dict | UUID): The content to send.
            identifier (UUID): The unique identifier for the session.
            session (ClientSession): The client's session.
        """
        data = AnyaData(
            dtype=dtype,
            content=raw_data,
            identifier=identifier,
            timestamp=datetime.now(timezone.utc),
        )
        try:
            packed_data = ormsgpack.packb(data, option=ormsgpack.OPT_SERIALIZE_PYDANTIC)
        except Exception as e:
            logger.exception(f"Failed to pack data: {e}")
            return

        try:
            await asyncio.wait_for(
                session.websocket.send(packed_data), timeout=session.timeout
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"[Server send to Client] Timeout after {session.timeout} secs, ignored"
            )
        except websockets.exceptions.ConnectionClosed:
            logger.warning(
                f"[Client] Connection lost: {session.websocket.remote_address}"
            )
            self.sessions.discard(session)
        except Exception as e:
            logger.exception(f"Unexpected error when sending to client: {e}")

    async def send_back_text(self, session: ClientSession):
        """
        Send back text response to the client.
        Ensures only one text response is being sent at a time.
        """
        if session.only_one_text_event.is_set():
            return
        session.only_one_text_event.set()
        try:
            unique_id, text = await session.llm_text_queue.get()
            await self.send_response(AnyaData.Type.TEXT, text, unique_id, session)
            session.text_first_event.set()  # Indicate that text has been sent

            if session.func_calling and not session.tool_call_queue.empty():
                unique_id, tool_call = await session.tool_call_queue.get()
                await self.send_response(
                    AnyaData.Type.EVENT, tool_call, unique_id, session
                )

        except Exception as e:
            logger.exception(f"Error in send_back_text: {e}")
        finally:
            session.only_one_text_event.clear()

    async def send_back_audio(self, session: ClientSession):
        """
        Send back audio response to the client.
        Ensures only one audio response is being sent at a time and waits for text to be sent first.
        """
        if session.only_one_audio_event.is_set():
            return
        session.only_one_audio_event.set()
        try:
            await session.text_first_event.wait()  # Wait until text is sent
            unique_id, audio = await session.tts_audio_queue.get()
            await self.send_response(AnyaData.Type.AUDIO, audio, unique_id, session)
        except Exception as e:
            logger.exception(f"Error in send_back_audio: {e}")
        finally:
            session.only_one_audio_event.clear()

    async def handle_client(self, websocket: ClientConnection, path: str = ""):
        """
        Handle incoming WebSocket client connections and process their data.

        Args:
            websocket (ClientConnection): The WebSocket connection.
            path (str): The path of the WebSocket connection.
        """
        client_address = websocket.remote_address
        logger.info(f"[Client connected]: {client_address}")
        session = ClientSession(
            websocket, self.asr, self.llm, self.tts, self.vad, self.memory_cls
        )
        self.sessions.add(session)
        try:
            # 启动会话的处理任务
            tasks = [
                asyncio.create_task(self.receive_data(session)),
                asyncio.create_task(self.process_input(session)),
                asyncio.create_task(self.respond_with_text(session)),
                asyncio.create_task(self.respond_with_audio(session)),
                asyncio.create_task(self.wait_cancel_task(session)),
            ]
            await asyncio.gather(*tasks)
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"Client {client_address} disconnected: {e}")
        except Exception as e:
            logger.exception(f"Error handling client {client_address}: {e}")
        finally:
            self.sessions.discard(session)
            logger.info(f"[Client disconnected]: {client_address}")

    async def receive_data(self, session: ClientSession):
        """
        Receive data from the client's websocket and process it.

        Args:
            session (ClientSession): The client's session.
        """
        async for raw_data in self._data_wrapper(session.websocket):
            await self.process_raw_data(raw_data, session)

    async def process_raw_data(self, raw_data: bytes, session: ClientSession):
        """
        Process raw data received from a client.

        Args:
            raw_data (bytes): The raw data received.
            session (ClientSession): The client's session.
        """
        try:
            data_obj: dict = ormsgpack.unpackb(raw_data)
            data = AnyaData(**data_obj)

            if data.dtype == AnyaData.Type.TEXT:
                await self.handle_text_input(data, session)
            elif data.dtype == AnyaData.Type.AUDIO:
                await self.handle_audio_input(data, session)
            elif data.dtype == AnyaData.Type.EVENT:
                await self.handle_event(data, session)
            else:
                logger.warning("Received unsupported AnyaData.Type")
        except Exception as e:
            logger.exception(f"Error processing raw data: {e}")

    async def handle_text_input(self, data: AnyaData, session: ClientSession):
        """
        Handle text input from the client.

        Args:
            data (AnyaData): The data object containing text.
            session (ClientSession): The client's session.
        """
        text = data.content.strip()
        if text:
            logger.info(f"[Text input]: {text}")
            session.accepted_identifier = data.identifier
            await session.text_input_queue.put((data.identifier, text))

    async def handle_audio_input(self, data: AnyaData, session: ClientSession):
        """
        Handle audio input from the client.

        Args:
            data (AnyaData): The data object containing audio.
            session (ClientSession): The client's session.
        """
        chunk = data.content
        try:
            for audio_bytes in session.vad.detect_speech(chunk):
                if audio_bytes == b"<|PAUSE|>":
                    await self._handle_cancel_event(session)
                    await self.send_response(
                        AnyaData.Type.EVENT,
                        AnyaData.Event.CANCEL,
                        data.identifier,
                        session,
                    )
                elif audio_bytes == b"<|RESUME|>":
                    await self._handle_resume(session)
                elif len(audio_bytes) > 1024:
                    # Detected audio activity (voice)
                    await self._handle_new_audio_input(
                        audio_bytes, data.identifier, session
                    )
        except Exception as e:
            logger.exception(f"Error handling audio input: {e}")

    async def _handle_new_audio_input(
        self, audio_bytes: bytes, playback_uuid: UUID, session: ClientSession
    ):
        """
        Handle new audio input by transcribing it and sending the transcription to the client.

        Args:
            audio_bytes (bytes): The audio data.
            playback_uuid (UUID): The identifier for the playback session.
            session (ClientSession): The client's session.
        """
        if session.asr:
            new_uuid = uuid.uuid4()
            voice_input = await session.asr.recognize_speech(audio_bytes)
            logger.info(f"[Voice input]: {voice_input}")
            await self.send_response(
                AnyaData.Type.EVENT,
                {"new_uuid": new_uuid, "transcription": voice_input},
                playback_uuid,
                session,
            )
            session.accepted_identifier = new_uuid
        else:
            logger.warning("ASR is not activated! Please restart with ASR activated!")

    async def handle_event(self, data: AnyaData, session: ClientSession):
        """
        Handle events sent by the client.

        Args:
            data (AnyaData): The data object containing the event.
            session (ClientSession): The client's session.
        """
        event = data.content
        if isinstance(event, str):
            event_mapping = {
                AnyaData.Event.ACCEPT_TEXT.value: (
                    lambda: self.send_back_text(session)
                ),
                AnyaData.Event.ACCEPT_AUDIO.value: (
                    lambda: self.send_back_audio(session)
                ),
                AnyaData.Event.CANCEL.value: (
                    lambda: self._handle_cancel_event(session)
                ),
            }
            handler = event_mapping.get(
                event, (lambda: self._handle_unknown_event(session))
            )
            asyncio.create_task(handler())

        if isinstance(event, dict):
            if chat_history := event.get("chat_history", None):
                asyncio.create_task(
                    self._handle_chat_context_event(chat_history, session)
                )

    async def _handle_cancel_event(self, session: ClientSession):
        session.cancel_event.set()

    async def _handle_unknown_event(self, session: ClientSession):
        logger.warning("Received unknown AnyaData.Event")

    async def _handle_chat_context_event(
        self, chat_history: List[Dict[str, str]], session: ClientSession
    ):
        """
        Handle chat context events by updating the session's memory.

        Args:
            chat_history (List[Dict[str, str]]): The chat history.
            session (ClientSession): The client's session.
        """
        logger.info("Handling chat context event")
        session.chat_history_messages = chat_history
        session.chat_history_events.set()

    async def send_response(
        self,
        dtype: AnyaData.Type,
        raw_data: str | bytes | dict | UUID,
        identifier: UUID,
        session: ClientSession,
    ):
        """
        Send a response to a specific client.

        Args:
            dtype (AnyaData.Type): The type of data (TEXT, AUDIO, EVENT).
            raw_data (str | bytes | dict | UUID): The content to send.
            identifier (UUID): The unique identifier for the session.
            session (ClientSession): The client's session.
        """
        data = AnyaData(
            dtype=dtype,
            content=raw_data,
            identifier=identifier,
            timestamp=datetime.now(timezone.utc),
        )
        try:
            packed_data = ormsgpack.packb(data, option=ormsgpack.OPT_SERIALIZE_PYDANTIC)
        except Exception as e:
            logger.exception(f"Failed to pack data: {e}")
            return

        try:
            await asyncio.wait_for(
                session.websocket.send(packed_data), timeout=session.timeout
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"[Server send to Client] Timeout after {session.timeout} secs, ignored"
            )
        except websockets.exceptions.ConnectionClosed:
            logger.warning(
                f"[Client] Connection lost: {session.websocket.remote_address}"
            )
            self.sessions.discard(session)
        except Exception as e:
            logger.exception(f"Unexpected error when sending to client: {e}")

    async def send_back_text(self, session: ClientSession):
        """
        Send back text response to the client.
        Ensures only one text response is being sent at a time.
        """
        if session.only_one_text_event.is_set():
            return
        session.only_one_text_event.set()
        try:
            unique_id, text = await session.llm_text_queue.get()
            await self.send_response(AnyaData.Type.TEXT, text, unique_id, session)
            session.text_first_event.set()  # Indicate that text has been sent

            if session.func_calling and not session.tool_call_queue.empty():
                unique_id, tool_call = await session.tool_call_queue.get()
                await self.send_response(
                    AnyaData.Type.EVENT, tool_call, unique_id, session
                )

        except Exception as e:
            logger.exception(f"Error in send_back_text: {e}")
        finally:
            session.only_one_text_event.clear()

    async def send_back_audio(self, session: ClientSession):
        """
        Send back audio response to the client.
        Ensures only one audio response is being sent at a time and waits for text to be sent first.
        """
        if session.only_one_audio_event.is_set():
            return
        session.only_one_audio_event.set()
        try:
            await session.text_first_event.wait()  # Wait until text is sent
            unique_id, audio = await session.tts_audio_queue.get()
            await self.send_response(AnyaData.Type.AUDIO, audio, unique_id, session)
        except Exception as e:
            logger.exception(f"Error in send_back_audio: {e}")
        finally:
            session.only_one_audio_event.clear()

    async def process_input(self, session: ClientSession):
        """
        Continuously process inputs from text and voice queues for a specific session.

        Args:
            session (ClientSession): The client's session.
        """
        while True:
            try:
                if not session.text_input_queue.empty():
                    unique_id, user_input = await session.text_input_queue.get()
                    await self.handle_input(
                        unique_id, user_input, session, is_voice=False
                    )
                    logger.info("[Text input]: Complete.")

                elif not session.voice_input_queue.empty():
                    unique_id, user_input = await session.voice_input_queue.get()
                    await self.handle_input(
                        unique_id, user_input, session, is_voice=True
                    )
                    logger.info("[Voice input]: Complete.")

                await asyncio.sleep(0.1)
            except Exception as e:
                logger.exception(f"Error in process_input: {e}")

    async def handle_input(
        self,
        unique_id: UUID,
        user_input: str,
        session: ClientSession,
        is_voice: bool = False,
    ):
        """
        Handle individual user input, distinguishing between commands and regular prompts.

        Args:
            unique_id (UUID): The unique identifier for the session.
            user_input (str): The input text from the user.
            session (ClientSession): The client's session.
            is_voice (bool): Whether the input is from voice.
        """
        maybe_command = user_input.strip()

        if maybe_command.startswith("/"):
            await self.handle_command(maybe_command, is_voice, session)
            return

        prompt = maybe_command
        await session.llm_prompt_queue.put((unique_id, prompt))

    async def handle_command(
        self, command: str, is_voice: bool, session: ClientSession
    ):
        """
        Handle specific commands from the user.

        Args:
            command (str): The command string.
            is_voice (bool): Whether the command is from voice input.
            session (ClientSession): The client's session.
        """
        command_handlers = {
            "/clear": lambda: self._clear_memory(session),
            "/history": lambda: self._show_history(session),
            "/help": lambda: self._show_help(session),
            "/interrupt": lambda: self._interrupt_processing(session),
            "/pause": lambda: self._handle_pause(session),
            "/resume": lambda: self._handle_resume(session),
        }

        handler = command_handlers.get(command, lambda: self._unknown_command(session))
        await handler()

    async def _clear_memory(self, session: ClientSession):
        """Clear the chatbot memory."""
        session.memory.clear()
        logger.info("Memory cleared.")

    async def _show_history(self, session: ClientSession):
        """Show the chatbot conversation history."""
        history = session.memory.retrieve_all()
        logger.info("=" * 10 + " Chat History " + "=" * 10)
        logger.info(history)

    async def _show_help(self, session: ClientSession):
        """Display available commands."""
        help_text = "/clear /history /help /interrupt /pause /resume"
        logger.info(help_text)

    async def _interrupt_processing(self, session: ClientSession):
        """Interrupt ongoing processing."""
        logger.info("Interrupt command received.")
        session.cancel_event.set()

    async def _handle_pause(self, session: ClientSession):
        """Handle pause event."""
        session.cancel_event.set()
        await self.send_response(
            AnyaData.Type.EVENT,
            AnyaData.Event.CANCEL,
            session.accepted_identifier,
            session,
        )

    async def _handle_resume(self, session: ClientSession):
        """Handle resume event."""
        session.continue_event.set()
        logger.info("Resume event received.")

    async def _unknown_command(self, session: ClientSession):
        """Handle unknown commands."""
        logger.warning("Unknown command received. Use /help for available commands.")

    async def clean_queued_data(self, session: ClientSession):
        """
        Clear all data from the queues of a specific session.

        Args:
            session (ClientSession): The client's session.
        """
        logger.warning("Cleaning queued data.")
        queues = [
            session.llm_text_queue,
            session.tool_call_queue,
            session.tts_text_queue,
            session.tts_audio_queue,
            session.text_input_queue,
            session.voice_input_queue,
            session.llm_prompt_queue,
        ]
        for queue in queues:
            while not queue.empty():
                queue.get_nowait()
                queue.task_done()

    async def wait_cancel_task(self, session: ClientSession):
        """
        Wait for cancellation events and handle graceful shutdown of tasks for a specific session.

        Args:
            session (ClientSession): The client's session.
        """
        while True:
            await session.cancel_event.wait()
            session.continue_event.clear()  # Prevent continuation
            logger.info("Cancellation event triggered.")
            await asyncio.sleep(0.2)  # Allow tasks to handle cancellation
            await self.clean_queued_data(session)
            await self.send_response(
                AnyaData.Type.TEXT, "", session.accepted_identifier, session
            )
            await self.send_response(
                AnyaData.Type.AUDIO, b"", session.accepted_identifier, session
            )
            session.cancel_event.clear()
            session.continue_event.set()
            logger.info("Cancellation handled and reset.")

    async def respond_with_text(self, session: ClientSession):
        """
        Continuously generate and send text responses using the LLM for a specific session.

        Args:
            session (ClientSession): The client's session.
        """
        while True:
            try:
                unique_id, prompt = await session.llm_prompt_queue.get()
                logger.info(f"Generating response for prompt: {prompt}")

                await session.continue_event.wait()

                done_event = asyncio.Event()

                async def generate_text():
                    try:
                        logger.info(
                            f"Generating NEED_CONTEXT for unique_id: {unique_id}"
                        )
                        await self.send_response(
                            AnyaData.Type.EVENT,
                            AnyaData.Event.NEED_CONTEXT,
                            unique_id,
                            session,
                        )
                        await session.chat_history_events.wait()
                        session.memory.messages = session.chat_history_messages.copy()
                        session.chat_history_messages = []
                        session.chat_history_events.clear()

                        async for text in self.llm.generate_response(
                            prompt, session.memory
                        ):
                            if session.cancel_event.is_set():
                                break
                            if text in {".", "。"}:
                                continue
                            session.memory.store("assistant", text, delta=True)
                            await session.llm_text_queue.put((unique_id, text))

                            # Clean text for TTS
                            tts_text = (
                                session.bracket_patterns.sub("", text).strip() or "."
                            )
                            await session.tts_text_queue.put((unique_id, tts_text))

                            # Handle function calling if enabled
                            if session.func_calling:
                                tool_call_str = ""
                                async for tool_call_text in self.llm.generate_response(
                                    text, session.memory, tool_choice="required"
                                ):
                                    if session.cancel_event.is_set():
                                        break
                                    tool_call_str += tool_call_text
                                if tool_call_str.startswith("set"):
                                    func_name, params = self.llm.parse_function_call(
                                        tool_call_str
                                    )
                                    tool_call = {
                                        "func_name": func_name,
                                        "params": params,
                                    }
                                    await session.tool_call_queue.put(
                                        (unique_id, tool_call)
                                    )
                    except asyncio.CancelledError:
                        logger.warning("Text generation cancelled.")
                    except Exception as e:
                        logger.exception(f"Error during text generation: {e}")
                    finally:
                        await self.send_response(
                            AnyaData.Type.EVENT,
                            AnyaData.ContextEvent(chat_history=session.memory.messages),
                            unique_id,
                            session,
                        )
                        done_event.set()
                        await session.llm_text_queue.put((unique_id, ""))

                gen_task = asyncio.create_task(generate_text())

                async def monitor_cancel():
                    while not session.cancel_event.is_set() and not done_event.is_set():
                        await asyncio.sleep(0.05)
                    if session.cancel_event.is_set():
                        gen_task.cancel()

                monitor_task = asyncio.create_task(monitor_cancel())
                await asyncio.gather(gen_task, monitor_task)
            except Exception as e:
                logger.exception(f"Error in respond_with_text: {e}")

    async def respond_with_audio(self, session: ClientSession):
        """
        Continuously generate and send audio responses using the TTS for a specific session.

        Args:
            session (ClientSession): The client's session.
        """
        while True:
            try:
                await session.continue_event.wait()
                unique_id, text_chunk = await session.tts_text_queue.get()

                if not text_chunk:
                    continue

                done_event = asyncio.Event()

                async def generate_audio():
                    try:
                        async for audio_chunk in self.tts.synthesize(text_chunk):
                            if session.cancel_event.is_set():
                                break
                            await session.tts_audio_queue.put((unique_id, audio_chunk))
                    except asyncio.CancelledError:
                        logger.warning("Audio generation cancelled.")
                    except Exception as e:
                        logger.exception(f"Error during audio synthesis: {e}")
                    finally:
                        done_event.set()
                        await session.tts_audio_queue.put((unique_id, b""))

                gen_task = asyncio.create_task(generate_audio())

                async def monitor_cancel():
                    while not session.cancel_event.is_set() and not done_event.is_set():
                        await asyncio.sleep(0.05)
                    if session.cancel_event.is_set():
                        gen_task.cancel()

                monitor_task = asyncio.create_task(monitor_cancel())
                await asyncio.gather(gen_task, monitor_task)
            except Exception as e:
                logger.exception(f"Error in respond_with_audio: {e}")

    async def start_server(self, host: str, port: int):
        """
        Start the WebSocket server and run all necessary tasks.

        Args:
            host (str): The host address to bind the server.
            port (int): The port number to bind the server.
        """
        logger.info("Initialization complete.")
        logger.info(f"Server running at ws://{host}:{port}")

        server = websockets.serve(
            self.handle_client,
            host,
            port,
        )

        try:
            async with server:
                await asyncio.Future()  # Run forever
        except Exception as e:
            logger.exception(f"Server encountered an error: {e}")

    async def chat(
        self,
        server_ws_host: str,
        server_ws_port: int,
    ):
        """
        Entry point to start the chatbot server.

        Args:
            server_ws_host (str): The WebSocket server host.
            server_ws_port (int): The WebSocket server port.
        """
        await self.start_server(server_ws_host, server_ws_port)
