from __future__ import annotations

from openminion.modules.brain.diagnostics.status import (
    coerce_phase_status,
    normalize_phase_status,
)


def test_coerce_phase_status_drops_negative_token_counters_in_fallback() -> None:
    status = coerce_phase_status(
        {
            "trace_id": "t-neg-tokens",
            "status_key": "analyzing",
            "total_input_tokens_used": -3,
            "total_output_tokens_used": -2,
            "total_tokens_used": -1,
        }
    )

    assert status.status_key == "analyzing"
    assert status.total_input_tokens_used is None
    assert status.total_output_tokens_used is None
    assert status.total_tokens_used is None


def test_coerce_phase_status_strips_and_preserves_valid_status_key() -> None:
    status = coerce_phase_status(
        {
            "trace_id": "t-ws",
            "status_key": " completed ",
            "label": "Done",
            "terminal": True,
        }
    )

    assert status.status_key == "completed"
    assert status.label == "Done"
    assert status.terminal is True


def test_coerce_phase_status_fallback_preserves_status_key_and_parses_bool_strings() -> (
    None
):
    status = coerce_phase_status(
        {
            "trace_id": "t-fallback",
            "status_key": "completed",
            "step_index": -1,
            "terminal": "false",
            "token_usage_estimated": "0",
        }
    )

    assert status.status_key == "completed"
    assert status.label == "Completed."
    assert status.terminal is False
    assert status.token_usage_estimated is False

    estimated = coerce_phase_status(
        {
            "trace_id": "t-fallback-true",
            "status_key": "executing",
            "step_index": -1,
            "terminal": "true",
            "token_usage_estimated": "1",
        }
    )
    assert estimated.status_key == "executing"
    assert estimated.terminal is True
    assert estimated.token_usage_estimated is True


def test_normalize_phase_status_handles_negative_bool_and_nonfinite_counters() -> None:
    status = normalize_phase_status(
        trace_id="t-norm-bad",
        source_phase="ACT",
        mode_step_index=-4,
        mode_step_total=True,
        payload={
            "step_index": -1,
            "step_total": float("inf"),
            "turn.llm_call_count": -2,
            "turn.llm_call_limit": False,
            "total_input_tokens_used": -10,
            "total_output_tokens_used": float("-inf"),
            "total_tokens_used": -5,
            "token_usage_estimated": "false",
        },
    )

    assert status.status_key == "executing"
    assert status.step_index is None
    assert status.step_total is None
    assert status.mode_step_index is None
    assert status.mode_step_total is None
    assert status.llm_call_count is None
    assert status.llm_call_limit is None
    assert status.total_input_tokens_used is None
    assert status.total_output_tokens_used is None
    assert status.total_tokens_used is None
    assert status.token_usage_estimated is False


def test_normalize_phase_status_preserves_zero_and_positive_counters() -> None:
    status = normalize_phase_status(
        trace_id="t-norm-good",
        source_phase="ACT",
        payload={
            "step_index": 0,
            "step_total": "4",
            "turn.llm_call_count": 2,
            "turn.llm_call_limit": 8,
            "total_tokens_used": 0,
            "token_usage_estimated": "yes",
        },
    )

    assert status.step_index == 0
    assert status.step_total == 4
    assert status.llm_call_count == 2
    assert status.llm_call_limit == 8
    assert status.total_tokens_used == 0
    assert status.token_usage_estimated is True
