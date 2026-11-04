import json
import time
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any
from typing import override

import httpx
from anthropic import Anthropic
from mistralai.client import Mistral
from ollama import Client

_BATCH_TRANSFER_ATTEMPTS = 3
_BATCH_TRANSFER_BACKOFF_SECONDS = (5.0, 15.0)
_BATCH_POLL_BACKOFF_SECONDS = (5.0, 15.0, 30.0, 60.0)


def _object_get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _retry_transport(action: str, call: Any) -> Any:
    for attempt in range(1, _BATCH_TRANSFER_ATTEMPTS + 1):
        try:
            return call()
        except httpx.TransportError as exc:
            if attempt == _BATCH_TRANSFER_ATTEMPTS:
                raise
            pause = _BATCH_TRANSFER_BACKOFF_SECONDS[attempt - 1]
            print(
                f"{action} failed ({type(exc).__name__}: {exc}); "
                f"retrying in {pause:g}s.",
                flush=True,
            )
            time.sleep(pause)


def _read_downloaded_file(download: Any) -> bytes:
    stream = _object_get(download, "stream")
    if stream is None:
        raise RuntimeError("Mistral batch download has no stream.")
    return b"".join(stream)


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _message_content_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def _parse_batch_jsonl(raw_bytes: bytes, source: str) -> list[dict]:
    responses: list[dict] = []
    for line_number, line in enumerate(
        raw_bytes.decode("utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            response_item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Invalid JSON in Mistral {source} at line {line_number}."
            ) from exc
        if not isinstance(response_item, dict):
            raise RuntimeError(
                f"Invalid response in Mistral {source} at line {line_number}."
            )

        custom_id = response_item.get("custom_id")
        if custom_id is None:
            raise RuntimeError(
                f"Missing custom_id in Mistral {source} at line {line_number}."
            )

        response = _as_dict(response_item.get("response"))
        body_raw = response.get("body", response_item.get("body"))
        body = _as_dict(body_raw)
        choices_raw = body.get("choices", [])
        choices = choices_raw if isinstance(choices_raw, list) else [choices_raw]
        first_choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        message = first_choice.get("message", {})
        content = _message_content_to_text(
            message.get("content") if isinstance(message, dict) else message
        )
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        error = response_item.get("error") or response.get("error")
        if not error and isinstance(body_raw, str) and not body:
            error = body_raw

        responses.append(
            {
                "custom_id": str(custom_id),
                "content": content,
                "error": error,
                "raw": response_item,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }
        )
    return responses


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

        self._client = Mistral(api_key=api_key, timeout_ms=120_000)
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

    def batch_chat(
        self,
        model: str,
        requests: list[dict],
        *,
        max_tokens: int = 128_000,
        poll_interval_seconds: int = 1,
    ) -> list[dict]:
        """Run a Mistral batch job.

        This reproduces the historical AP-HP Mistral batch format:
        system prompt + user prompt + assistant prefix.
        """
        if not requests:
            return []

        buffer = BytesIO()
        expected_ids: list[str] = []

        for idx, item in enumerate(requests):
            custom_id = str(item.get("custom_id", idx))
            if custom_id in expected_ids:
                raise ValueError(f"Duplicate Mistral batch custom_id: {custom_id}")
            expected_ids.append(custom_id)

            request = {
                "custom_id": custom_id,
                "body": {
                    "max_tokens": max_tokens,
                    "messages": [
                        {
                            "role": "system",
                            "content": item["system_prompt"],
                        },
                        {
                            "role": "user",
                            "content": item["user_prompt"],
                        },
                        {
                            "role": "assistant",
                            "content": item["prefix"],
                            "prefix": True,
                        },
                    ],
                },
            }

            buffer.write(json.dumps(request, ensure_ascii=False).encode("utf-8"))
            buffer.write(b"\n")

        input_file = _retry_transport(
            "Mistral batch upload",
            lambda: self._client.files.upload(
                file={
                    "file_name": "stream_aphp_batch.jsonl",
                    "content": buffer.getvalue(),
                },
                purpose="batch",
            ),
        )

        try:
            batch_job = self._client.batch.jobs.create(
                input_files=[input_file.id],
                model=model,
                endpoint="/v1/chat/completions",
                metadata={"job_type": "stream_aphp"},
            )
        except httpx.TransportError as exc:
            raise RuntimeError(
                "Mistral batch job creation failed with an uncertain state. "
                "Check active Mistral jobs before retrying."
            ) from exc

        poll_errors = 0
        while batch_job.status in ["QUEUED", "RUNNING"]:
            time.sleep(poll_interval_seconds)
            try:
                batch_job = self._client.batch.jobs.get(job_id=batch_job.id)
            except httpx.TransportError as exc:
                pause = _BATCH_POLL_BACKOFF_SECONDS[
                    min(poll_errors, len(_BATCH_POLL_BACKOFF_SECONDS) - 1)
                ]
                poll_errors += 1
                print(
                    f"Mistral batch {batch_job.id} polling failed "
                    f"({type(exc).__name__}: {exc}); retrying in {pause:g}s.",
                    flush=True,
                )
                time.sleep(pause)
                continue
            poll_errors = 0

        if batch_job.status not in {"SUCCESS", "SUCCEEDED"}:
            raise RuntimeError(
                f"Mistral batch job {batch_job.id} ended with status {batch_job.status}"
            )

        output_file_id = getattr(batch_job, "output_file", None) or getattr(
            batch_job, "output_file_id", None
        )
        error_file_id = getattr(batch_job, "error_file", None) or getattr(
            batch_job, "error_file_id", None
        )
        if output_file_id is None and error_file_id is None:
            raise RuntimeError(
                f"Mistral batch job {batch_job.id} has no output or error file."
            )

        responses: list[dict] = []
        for file_id, source in (
            (output_file_id, "output file"),
            (error_file_id, "error file"),
        ):
            if file_id is None:
                continue
            raw_bytes = _retry_transport(
                f"Mistral batch {source} download",
                lambda file_id=file_id: _read_downloaded_file(
                    self._client.files.download(file_id=file_id)
                ),
            )
            responses.extend(_parse_batch_jsonl(raw_bytes, source))

        received_ids = [response["custom_id"] for response in responses]
        duplicate_ids = sorted(
            {custom_id for custom_id in received_ids if received_ids.count(custom_id) > 1}
        )
        if duplicate_ids:
            raise RuntimeError(
                "Duplicate custom_id values in Mistral batch response: "
                + ", ".join(duplicate_ids)
            )
        if set(received_ids) != set(expected_ids):
            missing = sorted(set(expected_ids) - set(received_ids))
            unexpected = sorted(set(received_ids) - set(expected_ids))
            raise RuntimeError(
                "Mistral batch response identifiers do not match requests "
                f"(missing={missing}, unexpected={unexpected})."
            )

        responses_by_id = {response["custom_id"]: response for response in responses}
        return [responses_by_id[custom_id] for custom_id in expected_ids]


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
