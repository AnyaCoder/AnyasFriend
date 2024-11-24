# anyasfriend/components/interfaces/memory.py

from abc import ABC, abstractmethod


class Memory(ABC):
    @abstractmethod
    def store(self, role: str, content: str) -> None:
        """
        将每轮对话的消息存储到记忆中。
        :param role: 用户或助手
        :param content: 对话内容
        """
        pass

    @abstractmethod
    def retrieve_all(self) -> list:
        """
        获取所有对话历史，按顺序返回。
        :return: 存储的对话历史
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        清除所有存储的对话历史。
        """
        pass
