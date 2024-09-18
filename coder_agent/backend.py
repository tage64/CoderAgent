import json
from abc import ABC, abstractmethod

import requests
from groq import Groq
from openai import OpenAI


class Backend(ABC):
    """A backend for a large language model."""

    @abstractmethod
    def chat_completion(self, messages: list[dict[str, str]]) -> str:
        """Send a list of messages to the LLM and return the response."""


class GroqBackend(Backend):
    """A backend for Groq."""

    api_key: str = "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"
    model: str = "llama3-70b-8192"
    temperature: float = 0.0
    api_endpoint: str = "https://api.groq.com/openai/v1/chat/completions"

    def chat_completion(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            # Include other necessary parameters as per Groq API documentation
        }
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        response = requests.post(self.api_endpoint, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            result = response.json()
            # Adjust the parsing based on Groq's response format
            content = result["choices"][0]["message"]["content"]
            return content.strip()
        else:
            raise Exception(f"Groq API error: {response.status_code} - {response.text}")


class OpenaiBackend(Backend):
    """A backend for OpenAI."""

    client: OpenAI
    model: str = "gpt-4"
    max_tokens: int = 1500

    def __init__(self) -> None:
        self.client = OpenAI()

    def chat_completion(self, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore
            max_tokens=self.max_tokens,  # type: ignore
            temperature=temperature,
        )
        answer = response.choices[0].message.content or ""
        return answer.strip()


class GroqBackend2(Backend):
    """A backend for Groq."""

    client: Groq
    model: str = "llama3-70b-8192"
    max_tokens: int = 1500

    def __init__(self) -> None:
        self.client = Groq()

    def chat_completion(self, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore
            max_tokens=self.max_tokens,  # type: ignore
            temperature=temperature,
        )
        answer = response.choices[0].message.content or ""
        return answer.strip()
