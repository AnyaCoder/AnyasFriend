# main.py
import asyncio

import websockets
from loguru import logger

from anyasfriend.config import Config, config
from anyasfriend.factory import ChatbotFactory


async def main(config: Config):

    chatbot = ChatbotFactory.create_chatbot(config=config.chatbot)

    text_websocket = websockets.serve(
        chatbot.listen_for_text,
        config.backend.text_ws_host,
        config.backend.text_ws_port,
    )
    voice_websocket = websockets.serve(
        chatbot.listen_for_voice,
        config.backend.voice_ws_host,
        config.backend.voice_ws_port,
    )

    await asyncio.gather(
        text_websocket,
        voice_websocket,
        chatbot.chat(),
    )


if __name__ == "__main__":
    logger.info(f"{config.model_dump_json(indent=4)}")
    asyncio.run(main(config))
