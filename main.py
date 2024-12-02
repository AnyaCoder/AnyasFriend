# main.py
import asyncio

from loguru import logger

from anyasfriend.config import Config, config
from anyasfriend.factory import ChatbotFactory


async def main(config: Config):

    chatbot = ChatbotFactory.create_chatbot(config=config.chatbot)

    await chatbot.chat(
        config.backend.text_ws_host,
        config.backend.text_ws_port,
        config.backend.voice_ws_host,
        config.backend.voice_ws_port,
    )


if __name__ == "__main__":
    # handle Ctrl+C
    import signal

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    logger.info(f"{config.model_dump_json(indent=4)}")
    asyncio.run(main(config))
