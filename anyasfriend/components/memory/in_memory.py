# anyasfriend/components/memory/in_memory.py

from loguru import logger

from anyasfriend.components.interfaces.memory import Memory
from anyasfriend.config import config


class InMemory(Memory):
    def __init__(self):
        self.messages: list[dict[str, str]] = []
        self.set_system_message(config.chatbot.llm.system_message)

    def store(self, role: str, content: str, delta: bool = False) -> None:
        message = {"role": role, "content": content}

        if delta:
            assert len(self.messages) > 1
            if role != self.messages[-1]["role"]:
                self.store(role, content)
            else:
                self.messages[-1]["content"] += content
                logger.info(f"store(delta): {message}")
        else:
            logger.info(f"store: {message}")
            self.messages.append(message)

    def retrieve_all(self) -> list:
        return self.messages

    def clear(self) -> None:
        self.__init__()  # Re-initalize

    def set_system_message(self, system_message: str) -> None:
        if self.messages:
            assert self.messages[0]["role"] == "system"
            self.messages[0]["content"] = system_message
        else:
            self.store("system", system_message)
