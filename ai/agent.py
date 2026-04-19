import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from settings import settings

load_dotenv()


BASE_SYSTEM_PROMPT = (
    "You are a concise, reliable AI assistant for this application. "
    "Answer directly, use the provided context when available, and clearly state "
    "uncertainty instead of inventing facts."
)


@dataclass(slots=True)
class LlamaCppAgent:
    """
    Simple agent class that calls chat with system prompt.
    """

    client: OpenAI
    system_prompt: str = BASE_SYSTEM_PROMPT
    model: str = "Qwen-3-30B"  # NOTE: Does not matter
    default_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "temperature": 0.2,
            "max_tokens": 512,
        }
    )

    def invoke(self, user_input: str, **kwargs: Any) -> str:
        request_kwargs = {**self.default_kwargs, **kwargs}

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input},
            ],
            **request_kwargs,
        )
        return response.choices[0].message.content or ""

    def __call__(self, user_input: str, **kwargs: Any) -> str:
        return self.invoke(user_input, **kwargs)


def build_agent() -> LlamaCppAgent:
    """
    Factory function to create a LlamaCppAgent instance
    with settings from environment variables.
    """
    api_url = settings.api_url
    api_key = settings.api_key

    client = OpenAI(
        api_key=api_key,
        base_url=api_url,
        default_headers={
            "X-API-Key": api_key,
        },
    )

    return LlamaCppAgent(
        client=client,
        system_prompt=BASE_SYSTEM_PROMPT,
        model=os.getenv("MODEL_NAME", "local-model"),
    )


agent = build_agent()
