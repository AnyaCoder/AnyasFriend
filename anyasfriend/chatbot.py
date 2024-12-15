import asyncio
import sys
import uuid
from datetime import datetime
from typing import Set
from uuid import UUID

import ormsgpack
import websockets
from loguru import logger
from tqdm.asyncio import tqdm
from websockets.asyncio.client import ClientConnection

from anyasfriend.components.interfaces import ASR, LLM, TTS, VAD, Core, Memory
from anyasfriend.schema import AnyaData

# Configure logger
logger.remove()
logger.add(sys.stdout, level="INFO")


class Chatbot(Core):
    """
    Chatbot class that handles WebSocket connections, processes text and audio inputs,
    integrates ASR, LLM, TTS, VAD, and Memory components, and manages responses.
    """

    def __init__(self, *, asr: ASR, llm: LLM, tts: TTS, vad: VAD, memory: Memory):
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.vad = vad
        self.memory = memory

        # Queues for handling different types of inputs and outputs
        self.text_input_queue: asyncio.Queue = asyncio.Queue()
        self.voice_input_queue: asyncio.Queue = asyncio.Queue()
        self.llm_prompt_queue: asyncio.Queue = asyncio.Queue()
        self.tts_text_queue: asyncio.Queue = asyncio.Queue()
        self.tool_call_queue: asyncio.Queue = asyncio.Queue()
        self.temp_text_queue: asyncio.Queue = asyncio.Queue()
        self.tts_audio_queue: asyncio.Queue = asyncio.Queue()

        # Events for managing task synchronization
        self.cancel_event = asyncio.Event()
        self.continue_event = asyncio.Event()
        self.continue_event.set()

        self.only_one_text_event = asyncio.Event()
        self.only_one_audio_event = asyncio.Event()
        self.text_first_event = asyncio.Event()

        # Set of connected WebSocket clients
        self.clients: Set[ClientConnection] = set()

        # Unique identifier for the current session
        self.accepted_identifier: UUID = uuid.uuid4()

        # Timeout for sending responses
        self.timeout: float = 2.0

        # Function calling
        self.func_calling = self.llm.config.base.func_calling

    async def _data_wrapper(self, websocket: ClientConnection):
        """Asynchronous generator to yield data from the websocket."""
        async for data in websocket:
            yield data

    async def send_response(
        self, dtype: AnyaData.Type, raw_data: str | bytes | dict, identifier: UUID
    ):
        """
        Send a response to all connected clients.

        Args:
            dtype (AnyaData.Type): The type of data (TEXT or AUDIO).
            raw_data (str | bytes): The content to send.
            identifier (UUID): The unique identifier for the session.
        """
        data = AnyaData(
            dtype=dtype,
            content=raw_data,
            identifier=identifier,
            timestamp=datetime.now(),
        )
        packed_data = ormsgpack.packb(data, option=ormsgpack.OPT_SERIALIZE_PYDANTIC)
        for websocket in self.clients.copy():
            try:
                await asyncio.wait_for(
                    websocket.send(packed_data), timeout=self.timeout
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"[Server send to Client] Timeout after {self.timeout} secs, ignored"
                )
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"[Client] Connection lost: {websocket.remote_address}")
                self.clients.discard(websocket)
            except Exception as e:
                logger.exception(f"Unexpected error when sending to client: {e}")

    async def send_back_text(self):
        """
        Send back text response to the client.
        Ensures only one text response is being sent at a time.
        """
        if self.only_one_text_event.is_set():
            return
        self.only_one_text_event.set()
        try:
            unique_id, text = await self.tts_text_queue.get()
            await self.send_response(AnyaData.Type.TEXT, text, unique_id)
            self.text_first_event.set()  # Indicate that text has been sent
            if self.func_calling and not self.tool_call_queue.empty():
                unique_id, tool_call = await self.tool_call_queue.get()
                await self.send_response(AnyaData.Type.EVENT, tool_call, unique_id)

        except Exception as e:
            logger.exception(f"Error in send_back_text: {e}")
        finally:
            self.only_one_text_event.clear()

    async def send_back_audio(self):
        """
        Send back audio response to the client.
        Ensures only one audio response is being sent at a time and waits for text to be sent first.
        """
        if self.only_one_audio_event.is_set():
            return
        self.only_one_audio_event.set()
        try:
            await self.text_first_event.wait()  # Wait until text is sent
            unique_id, audio = await self.tts_audio_queue.get()
            await self.send_response(AnyaData.Type.AUDIO, audio, unique_id)
        except Exception as e:
            logger.exception(f"Error in send_back_audio: {e}")
        finally:
            self.only_one_audio_event.clear()

    async def handle_client(self, websocket: ClientConnection, path: str = ""):
        """
        Handle incoming WebSocket client connections and process their data.

        Args:
            websocket (ClientConnection): The WebSocket connection.
            path (str): The path of the WebSocket connection.
        """
        logger.info(f"[Client connected]: {websocket.remote_address}")
        self.clients.add(websocket)
        try:
            async for raw_data in tqdm(
                self._data_wrapper(websocket), desc="Incoming Data"
            ):
                await self.process_raw_data(raw_data)
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"Client {websocket.remote_address} disconnected: {e}")
        except Exception as e:
            logger.exception(f"Error handling client {websocket.remote_address}: {e}")
        finally:
            self.clients.discard(websocket)
            logger.info(f"[Client disconnected]: {websocket.remote_address}")

    async def process_raw_data(self, raw_data: bytes):
        """
        Process raw data received from a client.

        Args:
            raw_data (bytes): The raw data received.
        """
        try:
            data_obj: dict = ormsgpack.unpackb(raw_data)
            data = AnyaData(**data_obj)

            if data.dtype == AnyaData.Type.TEXT:
                await self.handle_text_input(data)
            elif data.dtype == AnyaData.Type.AUDIO:
                await self.handle_audio_input(data)
            elif data.dtype == AnyaData.Type.EVENT:
                await self.handle_event(data)
            else:
                logger.warning("Received unsupported AnyaData.Type")
        except Exception as e:
            logger.exception(f"Error processing raw data: {e}")

    async def handle_text_input(self, data: AnyaData):
        """Handle text input from the client."""
        text = data.content.strip()
        if text:
            logger.info(f"[Text input]: {text}")
            await self.text_input_queue.put((data.identifier, text))

    async def handle_audio_input(self, data: AnyaData):
        """Handle audio input from the client."""
        chunk = data.content
        try:
            for audio_bytes in self.vad.detect_speech(chunk):
                if audio_bytes == b"<|PAUSE|>":
                    self.cancel_event.set()
                elif audio_bytes == b"<|RESUME|>":
                    pass  # Implement resume logic if needed
                elif len(audio_bytes) > 1024:
                    # detected audio activity (voice)
                    new_uuid = uuid.uuid4()
                    await self.send_response(
                        AnyaData.Type.EVENT, AnyaData.Event.CANCEL, new_uuid
                    )
                    voice_input = await self.asr.recognize_speech(audio_bytes)
                    logger.info(f"[Voice input]: {voice_input}")
                    await self.voice_input_queue.put((new_uuid, voice_input))
        except Exception as e:
            logger.exception(f"Error handling audio input: {e}")

    async def handle_event(self, data: AnyaData):
        """Handle events sent by the client."""
        event = data.content
        if data.identifier != self.accepted_identifier:
            # Update the accepted identifier and reset events
            self.accepted_identifier = data.identifier
            self.text_first_event.clear()
            logger.debug(f"Updated accepted_identifier: {self.accepted_identifier}")

        match event:
            case AnyaData.Event.ACCEPT_TEXT.value:
                asyncio.create_task(self.send_back_text())
            case AnyaData.Event.ACCEPT_AUDIO.value:
                asyncio.create_task(self.send_back_audio())
            case AnyaData.Event.CANCEL.value:
                self.cancel_event.set()
            case _:
                logger.warning("Received unknown AnyaData.Event")

    async def process_input(self):
        """
        Continuously process inputs from text and voice queues.
        """
        while True:
            try:
                if not self.text_input_queue.empty():
                    unique_id, user_input = await self.text_input_queue.get()
                    await self.handle_input(unique_id, user_input)
                    logger.info("[Text input]: Complete.")

                elif not self.voice_input_queue.empty():
                    unique_id, user_input = await self.voice_input_queue.get()
                    await self.handle_input(unique_id, user_input, is_voice=True)
                    logger.info("[Voice input]: Complete.")

                await asyncio.sleep(0.1)
            except Exception as e:
                logger.exception(f"Error in process_input: {e}")

    async def handle_input(
        self, unique_id: UUID, user_input: str, is_voice: bool = False
    ):
        """
        Handle individual user input, distinguishing between commands and regular prompts.

        Args:
            unique_id (UUID): The unique identifier for the session.
            user_input (str): The input text from the user.
            is_voice (bool): Whether the input is from voice.
        """
        maybe_command = user_input.strip()

        if maybe_command.startswith("/"):
            await self.handle_command(maybe_command, is_voice)
            return

        prompt = maybe_command
        await self.llm_prompt_queue.put((unique_id, prompt))

    async def handle_command(self, command: str, is_voice: bool):
        """
        Handle specific commands from the user.

        Args:
            command (str): The command string.
            is_voice (bool): Whether the command is from voice input.
        """
        match command:
            case "/clear":
                self.memory.clear()
                logger.info("Memory cleared.")
            case "/history":
                history = self.memory.retrieve_all()
                logger.info("=" * 10 + " Chat History " + "=" * 10)
                logger.info(history)
            case "/help":
                help_text = "/clear /history /help /interrupt /pause /resume"
                logger.info(help_text)
            case "/interrupt":
                logger.info("Interrupt command received.")
                self.cancel_event.set()
            case _:
                logger.warning(
                    "Unknown command received. Use /help for available commands."
                )

    async def clean_queued_data(self):
        """Clear all data from the queues."""
        logger.warning("Cleaning queued data.")
        queues = [
            self.tts_text_queue,
            self.tool_call_queue,
            self.temp_text_queue,
            self.tts_audio_queue,
            self.text_input_queue,
            self.voice_input_queue,
            self.llm_prompt_queue,
        ]
        for queue in queues:
            while not queue.empty():
                queue.get_nowait()
                queue.task_done()

    async def wait_cancel_task(self):
        """
        Wait for cancellation events and handle graceful shutdown of tasks.
        """
        while True:
            await self.cancel_event.wait()
            self.continue_event.clear()  # Prevent continuation
            logger.info("Cancellation event triggered.")
            await asyncio.sleep(0.2)  # Allow tasks to handle cancellation
            await self.clean_queued_data()
            await self.send_response(AnyaData.Type.TEXT, "", self.accepted_identifier)
            await self.send_response(AnyaData.Type.AUDIO, b"", self.accepted_identifier)
            self.cancel_event.clear()
            self.continue_event.set()
            logger.info("Cancellation handled and reset.")

    async def respond_with_text(self):
        """
        Continuously generate and send text responses using the LLM.
        """
        while True:
            try:
                unique_id, prompt = await self.llm_prompt_queue.get()
                logger.debug(f"Generating response for prompt: {prompt}")
                await self.continue_event.wait()

                done_event = asyncio.Event()

                async def generate_text():
                    try:
                        async for text in self.llm.generate_response(prompt):
                            if self.cancel_event.is_set():
                                break
                            if text == "." or text == "。":
                                continue
                            self.memory.store("assistant", text, delta=True)
                            await self.tts_text_queue.put((unique_id, text))
                            await self.temp_text_queue.put((unique_id, text))

                            # If we have pre-defined tools
                            if self.func_calling:
                                tool_call_str = ""
                                async for tool_call_text in self.llm.generate_response(
                                    text, tool_choice="required"
                                ):
                                    if self.cancel_event.is_set():
                                        break
                                    tool_call_str += tool_call_text
                                if tool_call_str.startswith(
                                    "set"
                                ):  # tools_list: [set_emotion, ..]
                                    func_name, params = self.llm.parse_function_call(
                                        tool_call_str
                                    )
                                    tool_call = dict(func_name=func_name, params=params)
                                    # logger.debug("Tool: " + tool_call_text)
                                    await self.tool_call_queue.put(
                                        (unique_id, tool_call)
                                    )
                                del tool_call_str

                    except asyncio.CancelledError:
                        logger.warning("Text generation cancelled.")
                    except Exception as e:
                        logger.exception(f"Error during text generation: {e}")
                    finally:
                        done_event.set()
                        await self.tts_text_queue.put((unique_id, ""))

                gen_task = asyncio.create_task(generate_text())

                async def monitor_cancel():
                    while not self.cancel_event.is_set() and not done_event.is_set():
                        await asyncio.sleep(0.05)
                    if self.cancel_event.is_set():
                        gen_task.cancel()

                monitor_task = asyncio.create_task(monitor_cancel())
                await asyncio.gather(gen_task, monitor_task)
            except Exception as e:
                logger.exception(f"Error in respond_with_text: {e}")

    async def respond_with_audio(self):
        """
        Continuously generate and send audio responses using the TTS.
        """
        while True:
            try:
                await self.continue_event.wait()
                unique_id, text_chunk = await self.temp_text_queue.get()

                if not text_chunk:
                    continue

                done_event = asyncio.Event()

                async def generate_audio():
                    try:
                        async for audio_chunk in self.tts.synthesize(text_chunk):
                            if self.cancel_event.is_set():
                                break
                            await self.tts_audio_queue.put((unique_id, audio_chunk))
                    except asyncio.CancelledError:
                        logger.warning("Audio generation cancelled.")
                    except Exception as e:
                        logger.exception(f"Error during audio synthesis: {e}")
                    finally:
                        done_event.set()
                        await self.tts_audio_queue.put((unique_id, b""))

                gen_task = asyncio.create_task(generate_audio())

                async def monitor_cancel():
                    while not self.cancel_event.is_set() and not done_event.is_set():
                        await asyncio.sleep(0.05)
                    if self.cancel_event.is_set():
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

        await asyncio.gather(
            server,
            self.process_input(),
            self.wait_cancel_task(),
            self.respond_with_text(),
            self.respond_with_audio(),
        )

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
