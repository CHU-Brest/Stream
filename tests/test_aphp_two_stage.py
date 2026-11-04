import polars as pl
import pytest

from pipelines.aphp.pipeline import APHPPipeline
from pipelines.aphp.report import (
    generate_aphp_reports_mistral_batch,
    generate_aphp_reports_mistral_batch_two_stage,
)


class FakeBatchClient:
    def __init__(self, fail_summary_id: str | None = None):
        self.calls = []
        self.fail_summary_id = fail_summary_id

    def batch_chat(self, model, requests, *, max_tokens, poll_interval_seconds):
        self.calls.append(
            {
                "model": model,
                "requests": requests,
                "max_tokens": max_tokens,
                "poll_interval_seconds": poll_interval_seconds,
            }
        )
        is_summary = model == "summary-model"
        responses = []
        for request in reversed(requests):
            custom_id = request["custom_id"]
            error = (
                {"message": "summary failed"}
                if is_summary and custom_id == self.fail_summary_id
                else None
            )
            responses.append(
                {
                    "custom_id": custom_id,
                    "content": (
                        "" if error else f"{'SUMMARY' if is_summary else 'REPORT'}-{custom_id}"
                    ),
                    "error": error,
                    "raw": {"id": custom_id},
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                }
            )
        return responses


def two_stage_df():
    return pl.DataFrame(
        {
            "generation_id": ["g0", "g1"],
            "scenario": ["scenario 0", "scenario 1"],
            "user_prompt": ["USER 0", "USER 1"],
            "summary_system_prompt": ["SUMMARY SYSTEM 0", "SUMMARY SYSTEM 1"],
            "summary_prefix": ["", ""],
            "final_system_prompt": ["FINAL SYSTEM 0", "FINAL SYSTEM 1"],
            "final_prefix": ["PREFIX 0", "PREFIX 1"],
        }
    )


def one_stage_df():
    return pl.DataFrame(
        {
            "generation_id": ["g0", "g1", "g2"],
            "scenario": ["scenario 0", "scenario 1", "scenario 2"],
            "user_prompt": ["USER 0", "USER 1", "USER 2"],
            "system_prompt": ["SYSTEM 0", "SYSTEM 1", "SYSTEM 2"],
            "prefix": ["PREFIX 0", "PREFIX 1", "PREFIX 2"],
        }
    )


def test_two_stage_runs_two_batches_and_joins_by_custom_id(tmp_path):
    client = FakeBatchClient()

    result = generate_aphp_reports_mistral_batch_two_stage(
        two_stage_df(),
        client,
        summary_model="summary-model",
        report_model="report-model",
        output_dir=tmp_path,
        summary_max_tokens=8000,
        report_max_tokens=128000,
        poll_interval_seconds=2,
    )

    assert [(call["model"], call["max_tokens"]) for call in client.calls] == [
        ("summary-model", 8000),
        ("report-model", 128000),
    ]
    assert "SUMMARY-0" in client.calls[1]["requests"][0]["user_prompt"]
    assert result["summary"].to_list() == ["SUMMARY-0", "SUMMARY-1"]
    assert result["report"].to_list() == ["REPORT-0", "REPORT-1"]
    assert result["summary_total_tokens"].to_list() == [30, 30]
    assert list(tmp_path.glob("aphp_mistral_batch_two_stage_reports_2_*.parquet"))


def test_one_stage_splits_requests_and_keeps_one_output(tmp_path):
    client = FakeBatchClient()

    result = generate_aphp_reports_mistral_batch(
        one_stage_df(),
        client,
        "report-model",
        output_dir=tmp_path,
        batch_size=2,
    )

    assert [
        [request["custom_id"] for request in call["requests"]]
        for call in client.calls
    ] == [["0", "1"], ["2"]]
    assert result["report"].to_list() == ["REPORT-0", "REPORT-1", "REPORT-2"]
    assert len(list(tmp_path.glob("aphp_mistral_batch_reports_3_*.parquet"))) == 1


def test_two_stage_splits_summary_and_report_requests(tmp_path):
    client = FakeBatchClient()

    result = generate_aphp_reports_mistral_batch_two_stage(
        two_stage_df(),
        client,
        summary_model="summary-model",
        report_model="report-model",
        output_dir=tmp_path,
        batch_size=1,
    )

    assert [
        (call["model"], [request["custom_id"] for request in call["requests"]])
        for call in client.calls
    ] == [
        ("summary-model", ["0"]),
        ("summary-model", ["1"]),
        ("report-model", ["0"]),
        ("report-model", ["1"]),
    ]
    assert result["report"].to_list() == ["REPORT-0", "REPORT-1"]


def test_mistral_batch_size_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="mistral_batch_size"):
        generate_aphp_reports_mistral_batch(
            one_stage_df(),
            FakeBatchClient(),
            "report-model",
            output_dir=tmp_path,
            batch_size=0,
        )


def test_two_stage_skips_failed_summaries_in_report_batch(tmp_path):
    client = FakeBatchClient(fail_summary_id="1")

    result = generate_aphp_reports_mistral_batch_two_stage(
        two_stage_df(),
        client,
        summary_model="summary-model",
        report_model="report-model",
        output_dir=tmp_path,
    )

    assert [request["custom_id"] for request in client.calls[1]["requests"]] == ["0"]
    assert result["report"].to_list() == ["REPORT-0", ""]
    assert "summary failed" in result["summary_mistral_batch_error"][1]


def test_two_stage_requires_its_fictomed_columns(tmp_path):
    with pytest.raises(ValueError, match="final_prefix"):
        generate_aphp_reports_mistral_batch_two_stage(
            two_stage_df().drop("final_prefix"),
            FakeBatchClient(),
            summary_model="summary-model",
            report_model="report-model",
            output_dir=tmp_path,
        )


def test_pipeline_rejects_two_stage_direct_mode(tmp_path):
    pipeline = APHPPipeline(
        config={
            "data": {"output": str(tmp_path)},
            "generation": {"mode": "direct", "workflow": "two_stage"},
        },
        prompt={},
        servers={},
    )

    with pytest.raises(ValueError, match="mistral_batch"):
        pipeline.get_report(pl.DataFrame(), object(), "model")
