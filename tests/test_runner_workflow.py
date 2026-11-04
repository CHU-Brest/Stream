from pathlib import Path

import polars as pl

import runner


class DummyPipeline:
    def __init__(self, config, prompt, servers):
        self.servers = servers

    def get_report(self, df, client, model, batch_size):
        return df


def test_runner_forwards_prompt_configuration_to_fictomed(monkeypatch):
    calls = []
    config = {
        "pipelines": {
            "aphp": {
                "generation": {
                    "workflow": "two_stage",
                    "prompt_set": "old",
                    "prompt_dir": "/tmp/custom-prompts",
                },
            }
        },
        "servers": {},
    }

    monkeypatch.setitem(runner.PIPELINES, "aphp", DummyPipeline)
    monkeypatch.setattr(
        runner,
        "load_config",
        lambda path: {} if Path(path).name == "prompts.yaml" else config,
    )
    monkeypatch.setattr(
        runner,
        "generate",
        lambda **kwargs: calls.append(kwargs)
        or pl.DataFrame({"generation_id": [], "scenario": []}),
    )
    monkeypatch.setattr(runner, "get_client", lambda *_: (object(), "model"))

    runner.run("aphp", n_sejours=2)

    assert calls[0]["prompt_workflow"] == "two_stage"
    assert calls[0]["prompt_set"] == "old"
    assert calls[0]["prompt_dir"] == "/tmp/custom-prompts"
