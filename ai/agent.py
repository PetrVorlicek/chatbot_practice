import os
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from ai.semantic_store import ChunkMatch, SemanticStore
from settings import settings


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

    semantic_store: SemanticStore | None = None

    def _build_system_prompt(self, user_input: str) -> str:
        if self.semantic_store is None:
            return self.system_prompt

        try:
            matches = self.semantic_store.search(self.client, user_input, top_k=3)
        except Exception:
            return self.system_prompt

        if not matches:
            return self.system_prompt

        return compose_prompt_with_context(self.system_prompt, matches)

    def invoke(self, user_input: str, **kwargs: Any) -> str:
        request_kwargs = {**self.default_kwargs, **kwargs}
        system_prompt = self._build_system_prompt(user_input)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
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
    semantic_store = SemanticStore(settings.semantic_db_path)

    return LlamaCppAgent(
        client=client,
        system_prompt=BASE_SYSTEM_PROMPT,
        model=os.getenv("MODEL_NAME", "local-model"),
        semantic_store=semantic_store,
    )


def compose_prompt_with_context(system_prompt: str, matches: list[ChunkMatch]) -> str:
    sections = []
    for idx, match in enumerate(matches, start=1):
        sections.append(
            f"[{idx}] source={match.source_file}\n{match.text}"
        )

    context_block = "\n\n".join(sections)
    return (
        f"{system_prompt}\n\n"
        "Relevant support documentation excerpts:\n"
        f"{context_block}\n\n"
        "Use these excerpts when relevant. If they do not answer the question, say so clearly."
    )


agent = build_agent()
