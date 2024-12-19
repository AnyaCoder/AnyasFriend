import json
import re
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Optional

import httpx
from loguru import logger
from pydantic import BaseModel


class LLMBaseConfig(BaseModel):
    api_key: str = "YOUR_API_KEY"
    base_url: str = "http://localhost:11434"
    func_calling: bool = True


class AnyLLMConfig(BaseModel):
    base: LLMBaseConfig


class LLM(ABC):

    TERMINATORS = "。！？；，,.!?；"

    def __init__(self, config: AnyLLMConfig):
        self.config = config
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0, read=10.0, write=10.0),
            proxies={
                # no proxy for local
                "http://host.docker.internal": None,
                "http://127.0.0.1": None,
                "https://127.0.0.1": None,
                "http://localhost": None,
                "https://localhost": None,
            },
        )

    @abstractmethod
    async def generate_response(
        self, prompt: str, tool_choice: str
    ) -> AsyncGenerator[str, Any]:
        """
        基于输入的提示生成模型回复。
        :param prompt: 输入的提示文本
        :param tool_choice:
            控制模型调用 tool 的行为。
            `none` 意味着模型不会调用任何 tool，而是生成一条消息。
            `auto` 意味着模型可以选择生成一条消息或调用一个或多个 tool。
            `required` 意味着模型必须调用一个或多个 tool。
            通过 {"type": "function", "function": {"name": "my_function"}} 指定特定 tool，会强制模型调用该 tool。
            当没有 tool 时，默认值为 none。如果有 tool 存在，默认值为 auto。
        :yield: 生成的回复文本
        """
        pass

    @abstractmethod
    async def adjust_params(self, params: BaseModel) -> None:
        """
        异步地调整 LLM 参数
        """
        pass

    async def close(self) -> None:
        """关闭 httpx 客户端连接池"""
        await self.client.aclose()

    async def process_chunk_to_json(self, chunk: bytes):
        """将字节流切分并转换为 JSON 格式，结合流式处理器处理"""
        data_parts = chunk.split(b"\n\n")
        for data in data_parts:
            if not data.strip():
                continue

            if data.startswith(b"data: "):
                data = data[6:]

            if data.startswith(b"[DONE]"):
                break
            json_data = json.loads(data.decode("utf-8"))
            yield json_data

    def parse_function_call(self, text: str):
        # 正则表达式提取函数名和参数部分
        match = re.match(r"(\w+)\{(.*)\}", text.strip())

        if match:
            func_name = match.group(1)  # 获取函数名
            param_str = match.group(2)  # 获取参数部分（JSON格式）
            param_str = param_str.replace("'", '"')
            params = json.loads("{" + param_str + "}")

            return func_name, params
        else:
            raise ValueError("输入字符串格式错误")


class TextStreamProcessor:
    sentence_endings = re.compile(r"[。！？；：:.!?;~]")  # 句子结束符
    number_with_dot = re.compile(r".*\d+\..*$")  # 匹配以数字+句点结尾的部分

    def __init__(self):
        self.buffer: str = ""  # 用于存储未完成的句子

    async def process(
        self, text_stream: AsyncGenerator[str, Any]
    ) -> AsyncGenerator[str, Any]:
        """
        处理输入的文本流，确保每次返回完整的句子。
        :param text_stream: 输入的文本流，逐段传入
        :yield: 完整的句子
        """
        search_start = 0  # 记录当前的搜索起始位置

        async for text in text_stream:
            self.buffer += text.strip("\n")  # 将当前段落追加到缓冲区

            if search_start >= len(self.buffer):
                search_start = 0  # 重置搜索位置，准备处理下一个文本流

            # 查找文本中的句子结束符，并将完整句子提取出来
            while True:

                match = self.sentence_endings.search(self.buffer, search_start)
                if match:
                    # 找到完整句子结束符的位置
                    end_pos = match.end()
                    # print(self.buffer)
                    # print(match)
                    # print("start: ", search_start)
                    # 如果结束符前面是小数（例如 "3."），继续合并数据
                    if self.is_truncated_number(self.buffer[search_start:end_pos]):
                        # 如果是数字加句点，跳过当前结束符，继续处理后面的部分
                        # 更新搜索起始位置到当前句号后
                        search_start = end_pos
                        break

                    # 找到完整句子
                    sentence = self.buffer[:end_pos].strip()
                    if sentence:
                        search_start = 0  # reset
                        yield sentence

                    self.buffer = self.buffer[end_pos:].strip()
                else:
                    # 如果没有完整的句子，跳出，等待下一段流数据
                    break

        if self.buffer.strip():
            yield self.buffer

    def is_truncated_number(self, text):
        """判断是否是数字结尾（例如 "3."），表示该数字未完结"""
        return bool(self.number_with_dot.search(text))


async def main():
    text = """
这是一段包含数字和小数的文本。首先，有些简单的数字：42, 56, 89。然后是小数，如3.14、0.5、1.618。我们还可以看到一些复杂的数字格式，例如：123.45678、3.14159。
接着，我将加入一些描述性的句子，诸如"今天的温度是25.3度"。有时，数字后会有标点符号，如：0.5, 2.0，或者一些特殊字符：比如3.14!，还有数字3.0，甚至带有空格的 3.1415。

此外，还有一些不相关的文本，如字母和单词：a, b, c，以及包含句号、感叹号的句子。为了验证匹配的准确性，我们还会在文本末尾添加一些更复杂的数字，如：0.123、456.7890。
最后，文本中也包含了错误的小数格式，例如：123.，它缺少小数部分，或者空格分隔的小数，比如 3.14 跟随其后的数字 1。我们还会看到像 999.99 这样的数字组合。
测试结束！This is a test string that contains both English and 日本語; We will include some numbers like 12.34, 0.56, and 123.456.
For example, the number "3.14159" is often used in math; Meanwhile, Japanese sentences might include 今日は25.6度, or 東京は31.5度の気温です。
In addition, here's a sentence with a decimal number: "The price is 99.99 dollars." Another example would be "彼は5.6秒で走りました" (He ran in 5.6 seconds).
Additionally, we might have random English words like "apple", "banana", or "computer" mixed with Japanese characters: 本を読むことは楽しいです (Reading books is fun).
Don't forget about small decimal numbers such as 3.1 or larger ones like 1234.5678.

Lastly, there are some more random strings: 3.1415 is the value of pi, and I am still testing, 100.1 is another example.

    """

    async def example_text_stream():
        # 模拟一个异步文本流，每次返回一个部分的文本
        text_stream = [c for c in text]
        for part in text_stream:
            await asyncio.sleep(0.01)  # 模拟异步延迟
            yield part

    text_stream = example_text_stream()
    processor = TextStreamProcessor()
    async for sentence in processor.process(text_stream):
        print(f"完整句子：{sentence}")


# 测试代码
if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
