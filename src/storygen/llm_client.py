from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class LLMResponse:
    raw_text: str
    parsed_json: dict[str, Any]
    metadata: dict[str, Any]


class BaseLLMClient:
    def generate_structured(self, *, messages: list[dict[str, str]], json_schema: dict[str, Any]) -> LLMResponse:
        raise NotImplementedError


class OpenAILLMClient(BaseLLMClient):
    def __init__(self, llm_config: dict[str, Any]) -> None:
        self.llm_config = llm_config
        self.model = llm_config.get("model") or "gpt-4o-2024-08-06"
        self.api_key_env = llm_config.get("api_key_env", "OPENAI_API_KEY")
        self.temperature = float(llm_config.get("temperature", 0.0))
        self.max_output_tokens = int(llm_config.get("max_output_tokens", 800))
        self.timeout_seconds = float(llm_config.get("timeout_seconds", 30))

    def generate_structured(self, *, messages: list[dict[str, str]], json_schema: dict[str, Any]) -> LLMResponse:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing OpenAI API key environment variable: {self.api_key_env}")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("OpenAI SDK is missing. Install requirements.txt first.") from exc

        client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": self.timeout_seconds}
        api_base = self.llm_config.get("api_base")
        if api_base:
            client_kwargs["base_url"] = api_base

        client = OpenAI(**client_kwargs)
        response = client.responses.create(
            model=self.model,
            input=messages,
            text={
                "format": {
                    "type": "json_schema",
                    "name": json_schema["name"],
                    "schema": json_schema["schema"],
                    "strict": True,
                }
            },
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        raw_text = response.output_text
        return LLMResponse(
            raw_text=raw_text,
            parsed_json=json.loads(raw_text),
            metadata={
                "provider": "openai",
                "model": self.model,
                "response_id": getattr(response, "id", None),
            },
        )


def build_llm_client(llm_config: dict[str, Any]) -> BaseLLMClient:
    provider = llm_config.get("provider", "openai")
    if provider == "openai":
        return OpenAILLMClient(llm_config)
    raise ValueError(f"Unsupported LLM provider: {provider}")
