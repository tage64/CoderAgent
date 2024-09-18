from abc import ABC, abstractmethod

import dotenv
from groq import Groq
from openai import OpenAI

dotenv.load_dotenv(dotenv.find_dotenv())


class Backend(ABC):
    """A backend for a large language model."""

    @abstractmethod
    def chat_completion(self, messages: list[dict[str, str]]) -> str:
        """Send a list of messages to the LLM and return the response."""


class OpenaiBackend(Backend):
    """A backend for OpenAI."""

    client: OpenAI
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1500

    def __init__(self) -> None:
        self.client = OpenAI()

    def chat_completion(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore
            max_tokens=self.max_tokens,  # type: ignore
            temperature=self.temperature,
        )
        answer = response.choices[0].message.content or ""
        return answer.strip()


class GroqBackend(Backend):
    """A backend for Groq."""

    client: Groq
    model: str = "llama3-70b-8192"
    temperature: float = 0.0
    max_tokens: int = 1500

    def __init__(self) -> None:
        self.client = Groq()

    def chat_completion(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore
            max_tokens=self.max_tokens,  # type: ignore
            temperature=self.temperature,
        )
        answer = response.choices[0].message.content or ""
        return answer.strip()
