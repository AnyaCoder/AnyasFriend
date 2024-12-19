import json
from typing import Optional
from urllib.parse import urljoin

import httpx
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


class OllamaLLMRequestConfig(BaseModel):
    model: str = "qwen2.5:3b"
    stream: bool = True


class OllamaLLMConfig(AnyLLMConfig):
    request: OllamaLLMRequestConfig = OllamaLLMRequestConfig()


class OllamaLLMRequest(OllamaLLMRequestConfig):
    messages: list[dict[str, str]]
    tools: Optional[list[FunctionCallingTool]] = None
    tool_choice: Optional[str] = None


class OllamaLLM(LLM):
    def __init__(
        self,
        config: OllamaLLMConfig,
        memory: Memory,
    ):
        super().__init__(config)
        self.memory = memory
        if self.check_model_status():
            logger.info(f"OllamaLLM initialized! Current model: {config.request.model}")
        else:
            raise ValueError("Not OllamaLLM found!")

    async def generate_response(self, prompt: str, tool_choice: str = None):
        if tool_choice is None:
            self.memory.store(role="user", content=prompt)
        ollama_request = OllamaLLMRequest(
            model=self.config.request.model,
            stream=False if tool_choice is not None else True,
            messages=(
                self.memory.messages
                if tool_choice is None
                else [{"role": "user", "content": prompt}]
            ),
            tools=(
                None
                if tool_choice is None
                else [tool.model_dump() for tool in predefined_tools]
            ),
        )
        json_request = ollama_request.model_dump_json(indent=2, exclude_none=True)
        chat_url: str = urljoin(self.config.base.base_url, "api/chat")

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
                raise ValueError(
                    f"Failed to get response, status code: {response.status_code}"
                )
            if ollama_request.stream is False:
                response_text = b"".join(
                    [chunk async for chunk in response.aiter_bytes()]
                )
                response_json = json.loads(response_text.decode("utf-8"))
                if tool_choice:
                    delta: dict = response_json["message"]
                    if delta.get("tool_calls", None) is None:
                        return
                    func_chunk: dict = delta["tool_calls"][0]["function"]
                    tool_name_chunk = func_chunk.get("name", None)
                    if tool_name_chunk is not None:
                        yield str(tool_name_chunk)
                    tool_args_chunk = func_chunk.get("arguments", None)
                    if tool_args_chunk is not None:
                        yield str(tool_args_chunk)
                else:
                    assistant_reply: str = response_json["message"]["content"]
                    yield assistant_reply
            else:
                assistant_reply: str = ""

                async def textstream_generator():
                    async for chunk in response.aiter_bytes():
                        async for chunk_json in self.process_chunk_to_json(chunk):
                            chunk_text: str = chunk_json["message"]["content"]
                            yield chunk_text
                            if chunk_json["done"]:
                                break

                async for chunk_reply in stream_processor.process(
                    textstream_generator()
                ):
                    assistant_reply += chunk_reply
                    yield chunk_reply

            # self.memory.store("assistant", assistant_reply) # outside

    async def adjust_params(self, params: OllamaLLMConfig) -> None:
        self.config = params

    def check_model_status(self) -> bool:
        with httpx.Client(
            proxies={"http://localhost": None, "https://localhost": None}
        ) as client:
            test_url = urljoin(self.config.base.base_url, "api/ps")
            response = client.get(test_url)
            if response.status_code == 200:
                data: dict = response.json()
                models = data.get("models", [])
                if models:
                    for model in models:
                        name = model.get("name")
                        size = model.get("size")
                        expires_at = model.get("expires_at")
                        logger.debug(f"Model name: {name}")
                        logger.debug(f"Model size: {size} bytes")
                        logger.debug(f"Expire at: {expires_at}")
                        logger.info(f"{name} model is running.")
                        if name == self.config.request.model:
                            return True
                else:
                    logger.warning("No running model found.")
            else:
                logger.error(f"Request error, status: {response.status_code}")
        return False


async def llm_main():
    from anyasfriend.components.memory import InMemory
    from anyasfriend.config import config

    memory = InMemory()

    llm = OllamaLLM(
        config=OllamaLLMConfig(
            base=LLMBaseConfig(
                api_key=config.chatbot.llm.api_key,
                base_url=config.chatbot.llm.base_url,
            ),
            request=OllamaLLMRequestConfig(stream=True),
        ),
        memory=memory,
    )

    prompt = "给我讲一个笑话。"
    async for chunk_reply in llm.generate_response(prompt):
        logger.info(chunk_reply)


if __name__ == "__main__":
    import asyncio

    asyncio.run(llm_main())
