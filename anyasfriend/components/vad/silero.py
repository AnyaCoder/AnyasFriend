# anyasfriend/components/vad/silero_vad.py

import asyncio
from enum import Enum

import librosa
import numpy as np
import torch
from loguru import logger
from pydantic import BaseModel
from silero_vad import load_silero_vad

from anyasfriend.components.interfaces import VAD


class SileroVADConfig(BaseModel):
    orig_sr: int = 44100
    target_sr: int = 16000
    prob_threshold: float = 0.3
    db_threshold: int = 60


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
        audio_np_original = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        audio_np = audio_np_original / 32767
        audio_resampled_original = librosa.resample(
            audio_np_original,
            orig_sr=self.config.orig_sr,
            target_sr=self.config.target_sr,
        )
        audio_resampled = librosa.resample(
            audio_np, orig_sr=self.config.orig_sr, target_sr=self.config.target_sr
        )

        for i in range(0, len(audio_resampled), self.window_size_samples):
            chunk_np_original = audio_resampled_original[
                i : i + self.window_size_samples
            ]
            chunk_np = audio_resampled[i : i + self.window_size_samples]
            if len(chunk_np) < self.window_size_samples:
                break
            chunk = torch.Tensor(chunk_np)
            speech_prob = self.model(chunk, self.config.target_sr).item()

            if speech_prob:
                # print(f"Speech prob: {speech_prob * 100:.2f}%")
                iter = self.state.get_result(speech_prob, chunk_np_original)

                for prob, dbs, _bytes in iter:  # detected a sequence of voice bytes
                    rounded_probs = [round(x, 2) for x in prob]
                    rounded_dbs = [round(y, 2) for y in dbs]
                    # print("len: ", len(rounded_probs), ", probs: ", rounded_probs, " byte_len: ", len(_bytes))
                    # print("len: ", len(rounded_dbs), ", dbs: ", rounded_dbs)

                    yield _bytes


# 定义状态枚举
class State(Enum):
    IDLE = 1  # 空闲状态，等待语音
    ACTIVE = 2  # 语音检测状态
    INACTIVE = 3  # 语音结束状态（静默状态）


class StateMachine:
    def __init__(self, config: SileroVADConfig):
        self.state = State.IDLE  # 初始化为空闲状态
        self.prob_threshold = config.prob_threshold
        self.probs = []
        self.dbs = []
        self.bytes = bytes()
        self.miss_count = 0  # 计数器：连续未满足阈值的次数
        self.db_threshold = config.db_threshold

    def calculate_db(self, audio_data: np.ndarray) -> float:
        rms = np.sqrt(np.mean(np.square(audio_data)))
        reference = 1.0
        if rms == 0:
            return -np.inf
        db = 20 * np.log10(rms / reference)
        return db

    def update(self, chunk_bytes, prob, db):
        self.probs.append(prob)
        self.dbs.append(db)
        self.bytes += chunk_bytes

    def process(self, prob, chunk_np: np.ndarray):
        chunk_bytes = chunk_np.astype(np.int16).tobytes()
        db = self.calculate_db(chunk_np)

        if self.state == State.IDLE:
            if prob >= self.prob_threshold and db >= self.db_threshold:
                self.state = State.ACTIVE
                self.update(chunk_bytes, prob, db)
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
                self.miss_count = 0  # 重置计数器
            else:
                self.miss_count += 1  # 连续未满足阈值，计数器加1
                if self.miss_count >= 24:  # 连续24次不满足阈值 ~ 0.8 s 空音频
                    self.state = State.IDLE
                    if len(self.probs) > 30:  # ~ 1 s
                        yield self.probs.copy(), self.dbs.copy(), self.bytes
                    self.probs.clear()
                    self.dbs.clear()
                    self.bytes = b""
                    self.miss_count = 0  # 重置计数器

    def get_result(self, input_num, chunk_np):
        yield from self.process(input_num, chunk_np)


async def vad_main():
    global vad, audio_queue
    vad = SileroVAD(config=SileroVADConfig())
    audio_queue = asyncio.Queue()

    async def audio_handler(websocket):
        async for chunk in websocket:
            for _bytes in vad.detect_speech(chunk):
                await audio_queue.put(_bytes)

    async def start_websocket_server():
        import websockets

        host = "localhost"
        port = 8765
        start_server = websockets.serve(audio_handler, host, port)
        logger.info(f"WebSocket server started at ws://{host}:{port}")
        await start_server  # run forever until the task is cancelled

    await start_websocket_server()
    # await start_playback(audio_queue, sr=vad.config.target_sr)


if __name__ == "__main__":

    asyncio.run(vad_main())
