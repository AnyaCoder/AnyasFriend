# anyasfriend/components/llm/deepseek_llm.py

import json
from urllib.parse import urljoin

from loguru import logger
from pydantic import BaseModel

from anyasfriend.components.interfaces import LLM, AnyLLMConfig, LLMBaseConfig, Memory


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


class DeepSeekLLM(LLM):
    def __init__(
        self,
        config: DeepSeekLLMConfig,
        memory: Memory,
    ):
        super().__init__(config)
        self.memory = memory
        logger.info(f"DeepSeekLLM initialized!")

    async def generate_response(self, prompt: str):

        self.memory.store(role="user", content=prompt)

        json_request = DeepSeekLLMRequest(
            **self.config.request.model_dump(), messages=self.memory.messages
        ).model_dump_json(indent=4)

        chat_url: str = urljoin(self.config.base.base_url, "chat/completions")

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
                assistant_reply: str = ""

                async def textstream_generator():
                    async for chunk in response.aiter_bytes():
                        async for chunk_json in self.process_chunk_to_json(chunk):
                            chunk_text: str = chunk_json["choices"][0]["delta"][
                                "content"
                            ]
                            yield chunk_text

                async for chunk_reply in self.stream_processor.process(
                    textstream_generator()
                ):
                    assistant_reply += chunk_reply
                    yield chunk_reply

            # self.memory.store("assistant", assistant_reply) # outside

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
        memory=memory,
    )

    prompt = "给我讲一个笑话。"
    async for chunk_reply in llm.generate_response(prompt):
        logger.info(chunk_reply)


if __name__ == "__main__":
    import asyncio

    asyncio.run(llm_main())
