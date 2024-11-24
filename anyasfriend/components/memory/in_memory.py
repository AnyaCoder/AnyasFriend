# anyasfriend/components/memory/in_memory.py

from anyasfriend.components.interfaces.memory import Memory


class InMemory(Memory):
    def __init__(self):
        self.messages = []

    def store(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def retrieve_all(self) -> list:
        return self.messages

    def clear(self) -> None:
        self.messages = []
