from abc import ABC, abstractmethod
from typing import override

import httpx
from anthropic import Anthropic
from mistralai.client import Mistral
from ollama import Client


class BaseClient(ABC):
    """Base class for clients"""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def chat(self, model: str, messages: list[dict], **kwargs) -> dict:
        raise NotImplementedError


class AnthropicClient(BaseClient):
    """Anthropic (Claude) client normalised to the ``chat()`` interface.

    Parameters
    ----------
    api_key:
        Anthropic API key.
    verify:
        ``True`` (default), ``False``, or path to a CA bundle (``.pem``).
    """

    def __init__(self, api_key: str, verify: bool | str = True) -> None:
        super().__init__()

        self._client = Anthropic(
            api_key=api_key, http_client=httpx.Client(verify=verify)
        )

    @override
    def chat(self, model: str, messages: list[dict], **kwargs) -> dict:
        system: str | None = None
        chat_messages: list[dict] = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_messages.append(msg)

        params: dict = dict(
            model=model,
            max_tokens=kwargs.get("max_tokens", 4096),
            messages=chat_messages,
        )
        if system:
            params["system"] = system

        response = self._client.messages.create(**params)  # type: ignore[arg-type]

        return {
            "message": {
                "role": "assistant",
                "content": response.content[0].text,
            }
        }


class MistralClient(BaseClient):
    """Mistral client normalised to the ``chat()`` interface.

    Parameters
    ----------
    api_key:
        Mistral API key.
    verify:
        ``True`` (default), ``False``, or path to a CA bundle (``.pem``).
    """

    def __init__(self, api_key: str, verify: bool | str = True) -> None:
        super().__init__()

        self._client = Mistral(api_key=api_key)
        if verify is not True:
            self._client.client._client = httpx.Client(verify=verify)

    @override
    def chat(self, model: str, messages: list[dict], **kwargs) -> dict:
        response = self._client.chat.complete(
            model=model,
            max_tokens=kwargs.get("max_tokens", 4096),
            messages=messages,
        )

        return {
            "message": {
                "role": "assistant",
                "content": response.choices[0].message.content,
            }
        }


class OllamaClient(BaseClient):
    """Ollama client normalised to the ``chat()`` interface.
    Parameters
    ----------
    base_url:
        Ollama server URL (default: ``http://localhost:11434``).
    """

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        super().__init__()
        self._client = Client(host=base_url)

    @override
    def chat(self, model: str, messages: list[dict], **kwargs) -> dict:
        response = self._client.chat(
            model=model,
            messages=messages,
            options={"num_predict": kwargs.get("max_tokens", 4096)},
        )
        return {
            "message": {
                "role": "assistant",
                "content": response.message.content,
            }
        }


def get_client(
    servers: dict, client_type: str = "ollama"
) -> tuple[AnthropicClient | MistralClient | OllamaClient, str]:
    """Instantiate an LLM client and return ``(client, model_name)``."""
    if client_type == "ollama":
        cfg = servers["ollama"]
        return OllamaClient(cfg["host"]), cfg["model"]

    if client_type == "claude":
        cfg = servers["claude"]
        return AnthropicClient(
            api_key=cfg["api_key"],
            verify=cfg.get("verify", True),
        ), cfg["model"]

    if client_type == "mistral":
        cfg = servers["mistral"]
        return MistralClient(
            api_key=cfg["api_key"],
            verify=cfg.get("verify", True),
        ), cfg["model"]

    raise ValueError(
        f"Type de client inconnu : '{client_type}'. "
        "Valeurs acceptées : ollama, claude, mistral."
    )
