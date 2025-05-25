import asyncio
from typing import AsyncIterator

from openai import AsyncOpenAI

from ..config import settings
from ..interfaces.llm_base import LLM

client = AsyncOpenAI(api_key=settings.openai_api_key)
MODEL = "gpt-3.5-turbo"


class OpenAIChat(LLM):
    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """
        Stream the assistant’s reply token-by-token.
        """
        response = await client.chat.completions.create(
            model=MODEL,
            stream=True,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )

        async for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content
