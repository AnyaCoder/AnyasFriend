from pydantic import BaseModel

from anyasfriend.components.interfaces import LLM


class OpenAILLM(LLM):
    async def generate_response(self, prompt: str) -> str:
        return f"Response to '{prompt}' from OpenAI"

    async def adjust_params(self, params: BaseModel) -> None:
        pass

    async def close(self) -> None:
        await self.client.aclose()
