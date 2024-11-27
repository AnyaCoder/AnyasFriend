import json
from urllib.parse import urljoin

import httpx
from loguru import logger
from pydantic import BaseModel

from anyasfriend.components.interfaces import LLM, AnyLLMConfig, LLMBaseConfig, Memory


class OllamaLLMRequestConfig(BaseModel):
    model: str = "qwen2.5:3b"
    stream: bool = True


class OllamaLLMConfig(AnyLLMConfig):
    request: OllamaLLMRequestConfig = OllamaLLMRequestConfig()


class OllamaLLMRequest(OllamaLLMRequestConfig):
    messages: list[dict[str, str]]


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

    async def generate_response(self, prompt: str):

        self.memory.store(role="user", content=prompt)

        json_request = OllamaLLMRequest(
            **self.config.request.model_dump(), messages=self.memory.messages
        ).model_dump_json(indent=4)

        chat_url: str = urljoin(self.config.base.base_url, "api/chat")
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
            if self.config.request.stream is False:
                response_text = b"".join(
                    [chunk async for chunk in response.aiter_bytes()]
                )
                response_json = json.loads(response_text.decode("utf-8"))
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

                async for chunk_reply in self.stream_processor.process(
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
