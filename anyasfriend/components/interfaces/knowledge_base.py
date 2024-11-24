# anyasfriend/components/interfaces/knowledge_base.py

from abc import ABC, abstractmethod


class KnowledgeBase(ABC):
    @abstractmethod
    def query(self, query: str) -> str:
        """
        根据查询条件从知识库中获取相关信息。
        :param query: 查询条件
        :return: 查询结果
        """
        pass

    @abstractmethod
    def update(self, data: str) -> None:
        """
        向知识库中添加新的知识。
        :param data: 新的知识数据
        """
        pass
