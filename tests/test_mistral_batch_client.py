import json
from types import SimpleNamespace

import httpx
import pytest

import core.clients as clients
from core.clients import MistralClient


def batch_request(custom_id="0"):
    return {
        "custom_id": custom_id,
        "system_prompt": "SYSTEM",
        "user_prompt": "USER",
        "prefix": "PREFIX",
    }


def response_bytes(custom_id="0"):
    item = {
        "custom_id": custom_id,
        "response": json.dumps(
            {
                "body": json.dumps(
                    {
                        "choices": {
                            "message": {
                                "content": [
                                    {"type": "text", "text": "generated "},
                                    {"type": "text", "text": "report"},
                                ]
                            }
                        },
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 20,
                            "total_tokens": 30,
                        },
                    }
                )
            }
        ),
    }
    return (json.dumps(item) + "\n").encode()


class FakeFiles:
    def __init__(self, raw_bytes, *, fail_upload=False, fail_download=False):
        self.raw_bytes = raw_bytes
        self.fail_upload = fail_upload
        self.fail_download = fail_download
        self.upload_calls = 0
        self.download_calls = 0

    def upload(self, **kwargs):
        self.upload_calls += 1
        if self.fail_upload and self.upload_calls == 1:
            raise httpx.ConnectError("upload unavailable")
        return SimpleNamespace(id="input-file")

    def download(self, **kwargs):
        self.download_calls += 1
        if self.fail_download and self.download_calls == 1:
            raise httpx.ConnectError("download unavailable")
        return SimpleNamespace(stream=[self.raw_bytes])


class FakeJobs:
    def __init__(self, *, fail_poll=False, fail_create=False):
        self.fail_poll = fail_poll
        self.fail_create = fail_create
        self.create_calls = 0
        self.get_calls = 0

    def create(self, **kwargs):
        self.create_calls += 1
        if self.fail_create:
            raise httpx.ConnectError("creation response lost")
        return SimpleNamespace(id="job-1", status="RUNNING")

    def get(self, **kwargs):
        self.get_calls += 1
        if self.fail_poll and self.get_calls == 1:
            raise httpx.ConnectError("poll unavailable")
        return SimpleNamespace(
            id="job-1",
            status="SUCCESS",
            output_file="output-file",
            error_file=None,
        )


def fake_client(files, jobs):
    client = object.__new__(MistralClient)
    client._client = SimpleNamespace(
        files=files,
        batch=SimpleNamespace(jobs=jobs),
    )
    return client


def test_batch_chat_retries_transfers_and_polling(monkeypatch):
    files = FakeFiles(
        response_bytes(),
        fail_upload=True,
        fail_download=True,
    )
    jobs = FakeJobs(fail_poll=True)
    client = fake_client(files, jobs)
    sleeps = []
    monkeypatch.setattr(clients.time, "sleep", sleeps.append)

    responses = client.batch_chat(
        model="model",
        requests=[batch_request()],
        poll_interval_seconds=0,
    )

    assert files.upload_calls == 2
    assert jobs.create_calls == 1
    assert jobs.get_calls == 2
    assert files.download_calls == 2
    assert responses[0]["content"] == "generated report"
    assert responses[0]["total_tokens"] == 30
    assert sleeps == [5.0, 0, 5.0, 0, 5.0]


def test_batch_job_creation_is_not_retried():
    files = FakeFiles(response_bytes())
    jobs = FakeJobs(fail_create=True)
    client = fake_client(files, jobs)

    with pytest.raises(RuntimeError, match="uncertain state"):
        client.batch_chat(model="model", requests=[batch_request()])

    assert jobs.create_calls == 1


def test_batch_chat_rejects_missing_response_ids(monkeypatch):
    files = FakeFiles(response_bytes(custom_id="1"))
    jobs = FakeJobs()
    client = fake_client(files, jobs)
    monkeypatch.setattr(clients.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="identifiers do not match"):
        client.batch_chat(model="model", requests=[batch_request(custom_id="0")])


def test_batch_chat_rejects_duplicate_request_ids():
    client = fake_client(FakeFiles(response_bytes()), FakeJobs())

    with pytest.raises(ValueError, match="Duplicate"):
        client.batch_chat(
            model="model",
            requests=[batch_request("0"), batch_request("0")],
        )


def test_batch_chat_reports_invalid_json(monkeypatch):
    files = FakeFiles(b"not-json\n")
    jobs = FakeJobs()
    client = fake_client(files, jobs)
    monkeypatch.setattr(clients.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="Invalid JSON"):
        client.batch_chat(model="model", requests=[batch_request()])
