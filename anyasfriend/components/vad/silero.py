# anyasfriend/components/vad/silero_vad.py

import asyncio
from enum import Enum

import numpy as np
import torch
from loguru import logger
from pydantic import BaseModel
from silero_vad import load_silero_vad

from anyasfriend.components.interfaces import VAD


class SileroVADConfig(BaseModel):
    orig_sr: int = 16000
    target_sr: int = 16000
    prob_threshold: float = 0.3
    db_threshold: int = 40


class SileroVAD(VAD):
    def __init__(self, config: SileroVADConfig):
        self.config = config
        self.model = self.load_vad_model()
        self.state = StateMachine(config)
        self.window_size_samples = 512 if config.target_sr == 16000 else 256

    def load_vad_model(self):
        logger.info("Loading silero-VAD model...")
        return load_silero_vad()

    def detect_speech(self, audio_data: bytes):
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767
        for i in range(0, len(audio_np), self.window_size_samples):
            chunk_np = audio_np[i : i + self.window_size_samples]
            if len(chunk_np) < self.window_size_samples:
                break
            chunk = torch.Tensor(chunk_np)

            with torch.no_grad():
                speech_prob = self.model(chunk, self.config.target_sr).item()

            if speech_prob:
                # print(speech_prob)
                iter = self.state.get_result(speech_prob, chunk_np)

                for probs, dbs, chunk in iter:  # detected a sequence of voice bytes
                    rounded_probs = [round(x, 2) for x in probs]
                    rounded_dbs = [round(y, 2) for y in dbs]
                    print(
                        "len: ",
                        len(rounded_probs),
                        ", probs: ",
                        rounded_probs,
                        " byte_len: ",
                        len(chunk),
                    )
                    print("len: ", len(rounded_dbs), ", dbs: ", rounded_dbs)
                    audio_chunk = bytes(chunk)
                    yield audio_chunk

        del audio_np


# 定义状态枚举
class State(Enum):
    IDLE = 1  # 空闲状态，等待语音
    ACTIVE = 2  # 语音检测状态
    INACTIVE = 3  # 语音结束状态（静默状态）


class StateMachine:
    def __init__(self, config: SileroVADConfig):
        self.state = State.IDLE
        self.prob_threshold = config.prob_threshold
        self.probs = []
        self.dbs = []
        self.bytes = bytearray()
        self.miss_count = 0
        self.db_threshold = config.db_threshold

    @classmethod
    def calculate_db(cls, audio_data: np.ndarray) -> float:
        rms = np.sqrt(np.mean(np.square(audio_data)))
        return 20 * np.log10(rms) if rms > 0 else -np.inf

    def update(self, chunk_bytes, prob, db):
        self.probs.append(prob)
        self.dbs.append(db)
        self.bytes.extend(chunk_bytes)

    def reset_buffers(self):
        self.probs.clear()
        self.dbs.clear()
        self.bytes.clear()

    def process(self, prob, float_chunk_np: np.ndarray):
        int_chunk_np = float_chunk_np * 32767
        chunk_bytes = int_chunk_np.astype(np.int16).tobytes()
        db = self.calculate_db(int_chunk_np)

        if self.state == State.IDLE:
            if prob >= self.prob_threshold and db >= self.db_threshold:
                self.state = State.ACTIVE
                self.update(chunk_bytes, prob, db)
                yield [], [], b"<|PAUSE|>"
            else:
                pass
        elif self.state == State.ACTIVE:
            self.update(chunk_bytes, prob, db)
            if prob < self.prob_threshold:
                self.state = State.INACTIVE
            else:
                pass

        elif self.state == State.INACTIVE:
            self.update(chunk_bytes, prob, db)
            if prob >= self.prob_threshold:
                self.state = State.ACTIVE
                self.miss_count = 0
            else:
                self.miss_count += 1
                if self.miss_count >= 24:
                    self.state = State.IDLE
                    yield [], [], b"<|RESUME|>"
                    if len(self.probs) > 30:
                        yield self.probs, self.dbs, self.bytes
                        self.reset_buffers()
                    self.miss_count = 0

    def get_result(self, input_num, chunk_np):
        yield from self.process(input_num, chunk_np)


async def vad_main():
    global vad, audio_queue
    vad = SileroVAD(config=SileroVADConfig())
    audio_queue = asyncio.Queue()
    from tqdm.asyncio import tqdm

    async def data_wrapper(websocket):
        async for chunk in websocket:
            yield chunk

    async def audio_handler(websocket):
        async for chunk in tqdm(data_wrapper(websocket), desc="Audio chunk"):
            # print(len(chunk))
            for _bytes in vad.detect_speech(chunk):
                print(_bytes[:44])
                # await audio_queue.put(_bytes)
                pass

    async def empty_run():
        while True:
            await asyncio.sleep(0.1)

    async def start_websocket_server():
        import websockets

        host = "localhost"
        port = 8765
        start_server = websockets.serve(audio_handler, host, port)
        logger.info(f"WebSocket server started at ws://{host}:{port}")
        await start_server  # run forever until the task is cancelled

    await asyncio.gather(start_websocket_server(), empty_run())
    # await start_playback(audio_queue, sr=vad.config.target_sr)


if __name__ == "__main__":

    asyncio.run(vad_main())
