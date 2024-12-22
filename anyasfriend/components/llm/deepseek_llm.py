# anyasfriend/components/llm/deepseek_llm.py

import json
from typing import Optional
from urllib.parse import urljoin

from loguru import logger
from pydantic import BaseModel

from anyasfriend.components.interfaces import (
    LLM,
    AnyLLMConfig,
    LLMBaseConfig,
    Memory,
    TextStreamProcessor,
)
from anyasfriend.components.llm.func_call import Tool as FunctionCallingTool
from anyasfriend.components.llm.func_call import predefined_tools


class DeepSeekLLMRequestConfig(BaseModel):
    model: str = "deepseek-chat"
    frequency_penalty: int = 0
    max_tokens: int = 2048
    temperature: int = 1
    top_p: int = 1
    stream: bool = True


class DeepSeekLLMConfig(AnyLLMConfig):
    request: DeepSeekLLMRequestConfig = DeepSeekLLMRequestConfig()


class DeepSeekLLMRequest(DeepSeekLLMRequestConfig):
    messages: list[dict[str, str]]
    tools: Optional[list[FunctionCallingTool]] = None
    tool_choice: Optional[str] = None


class DeepSeekLLM(LLM):
    def __init__(
        self,
        config: DeepSeekLLMConfig,
    ):
        super().__init__(config)
        logger.info(f"DeepSeekLLM initialized!")

    async def generate_response(
        self, prompt: str, memory: Memory, tool_choice: str = None
    ):

        if tool_choice is None:
            memory.store(role="user", content=prompt)

        json_request = DeepSeekLLMRequest(
            **self.config.request.model_dump(),
            messages=(
                memory.messages
                if tool_choice is None
                else [{"role": "assistant", "content": prompt}]
            ),
            tools=(
                None
                if tool_choice is None
                else [tool.model_dump() for tool in predefined_tools]
            ),
            tool_choice=tool_choice,
        ).model_dump_json(indent=2, exclude_none=True)

        chat_url: str = urljoin(self.config.base.base_url, "chat/completions")

        stream_processor = TextStreamProcessor()

        async with self.client.stream(
            method="POST",
            url=chat_url,
            data=json_request,
            headers={
                "Authorization": f"Bearer {self.config.base.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        ) as response:
            if response.status_code != 200:
                logger.error(
                    f"Failed to get response, status code: {response.status_code}"
                )
                raise
            if self.config.request.stream is False:
                response_text = b"".join(
                    [chunk async for chunk in response.aiter_bytes()]
                )
                response_json = json.loads(response_text.decode("utf-8"))
                assistant_reply: str = response_json["choices"][0]["message"]["content"]
                yield assistant_reply
            else:
                # assistant_reply: list = []

                async def textstream_generator():
                    async for chunk in response.aiter_bytes():
                        async for chunk_json in self.process_chunk_to_json(chunk):
                            delta: dict = chunk_json["choices"][0]["delta"]
                            text_chunk = delta.get("content", None)
                            if text_chunk is not None:
                                yield text_chunk
                            else:
                                func_chunk: dict = delta["tool_calls"][0]["function"]
                                tool_name_chunk = func_chunk.get("name", None)
                                if tool_name_chunk is not None:
                                    yield tool_name_chunk
                                tool_args_chunk = func_chunk.get("arguments", None)
                                if tool_args_chunk is not None:
                                    yield tool_args_chunk

                async for chunk_reply in stream_processor.process(
                    textstream_generator()
                ):
                    # assistant_reply.append(chunk_reply)
                    yield chunk_reply

            # memory.store("assistant", "".join(assistant_reply)) # outside

    async def adjust_params(self, params: DeepSeekLLMConfig) -> None:
        self.config = params


async def llm_main():
    from anyasfriend.components.memory import InMemory

    memory = InMemory()
    memory.store("system", "You are a helpful assistant.")
    from anyasfriend.config import config

    llm = DeepSeekLLM(
        config=DeepSeekLLMConfig(
            base=LLMBaseConfig(
                api_key=config.chatbot.llm.api_key,
                base_url=config.chatbot.llm.base_url,
            ),
            request=DeepSeekLLMRequestConfig(stream=True),
        ),
    )

    prompt = "怎么办？我明明都会做啊，还是错的"
    async for chunk_reply in llm.generate_response(
        prompt, memory, tool_choice="required"
    ):
        logger.info(chunk_reply)


if __name__ == "__main__":
    import asyncio

    asyncio.run(llm_main())
