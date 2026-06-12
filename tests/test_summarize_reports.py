import pytest
import pandas as pd

from scripts.summarize_reports import (
    layer_control_gaps,
    paired_patching_effects,
    read_csv,
)


def test_layer_control_gaps_orders_best_layer_first():
    rows = [
        {"layer": 1, "condition": "mapping", "score": 0.8},
        {"layer": 1, "condition": "mention", "score": 0.2},
        {"layer": 1, "condition": "negation", "score": 0.1},
        {"layer": 2, "condition": "mapping", "score": 0.6},
        {"layer": 2, "condition": "mention", "score": 0.5},
        {"layer": 2, "condition": "negation", "score": 0.4},
    ]
    result = layer_control_gaps(
        pd.DataFrame(rows),
        "score",
        controls=["mention", "negation"],
    )
    assert result.iloc[0]["layer"] == 1
    assert result.iloc[0]["mean_control_gap"] == pytest.approx(0.65)


def test_layer_control_gaps_keeps_positions_separate():
    rows = [
        {"position": "query_source", "layer": 1, "condition": "mapping", "score": 0.6},
        {"position": "query_source", "layer": 1, "condition": "mention", "score": 0.5},
        {"position": "final_pre_answer", "layer": 1, "condition": "mapping", "score": 0.9},
        {"position": "final_pre_answer", "layer": 1, "condition": "mention", "score": 0.1},
    ]
    result = layer_control_gaps(
        pd.DataFrame(rows),
        "score",
        controls=["mention"],
    )
    assert result.iloc[0]["position"] == "final_pre_answer"
    assert len(result) == 2


def test_paired_patching_effects_uses_matching_unpatched_rows():
    rows = [
        {
            "example_id": "e1",
            "pair_id": "p1",
            "template_id": 2,
            "layer": 7,
            "intervention": "unpatched",
            "alpha": 0.0,
            "target_pref_logit": 2.0,
            "p_target_vs_source": 0.88,
        },
        {
            "example_id": "e1",
            "pair_id": "p1",
            "template_id": 2,
            "layer": 7,
            "intervention": "subtract_train_mean_delta",
            "alpha": 1.0,
            "target_pref_logit": 0.5,
            "p_target_vs_source": 0.62,
        },
    ]
    result = paired_patching_effects(pd.DataFrame(rows))
    assert result.iloc[0]["mean_target_pref_change"] == -1.5


def test_read_csv_skips_empty_outputs(tmp_path):
    (tmp_path / "probe.csv").write_text("", encoding="utf-8")
    assert read_csv(tmp_path, "probe.csv") is None
