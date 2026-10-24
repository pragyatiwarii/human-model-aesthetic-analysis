from app.services.agreement_analysis import (
    _average_ranks,
    analyze_batch_handoff,
    dedupe_preferences,
    summarize_dashboard,
)


def _handoff(model_scores=(4.6, 4.0, 3.4, 2.8), human_rows=None):
    ids = ["original:o1", "cf-s", "cf-m", "cf-x"]
    if human_rows is None:
        human_rows = [
            {
                "participant_code": "P001",
                "ratings": {ids[0]: 5, ids[1]: 4, ids[2]: 3, ids[3]: 2},
                "preferred_candidate_id": ids[0],
            },
            {
                "participant_code": "P002",
                "ratings": {ids[0]: 4, ids[1]: 4, ids[2]: 3, ids[3]: 2},
                "preferred_candidate_id": ids[0],
            },
            {
                "participant_code": "P003",
                "ratings": {ids[0]: 5, ids[1]: 4, ids[2]: 2, ids[3]: 1},
                "preferred_candidate_id": ids[0],
            },
        ]
    return {
        "batch_id": "batch-1",
        "original": {"dataset_image_name": "x.jpg", "prompt": "test", "category": "realistic"},
        "region": {"label": "person", "is_fallback": False},
        "perturbation": {"type": "blur", "label": "Gaussian blur"},
        "candidates": [
            {"candidate_id": ids[0], "source_type": "original", "severity": None, "image_url": "/o", "model_aesthetic_score": model_scores[0]},
            {"candidate_id": ids[1], "source_type": "counterfactual", "severity": "subtle", "image_url": "/s", "model_aesthetic_score": model_scores[1]},
            {"candidate_id": ids[2], "source_type": "counterfactual", "severity": "moderate", "image_url": "/m", "model_aesthetic_score": model_scores[2]},
            {"candidate_id": ids[3], "source_type": "counterfactual", "severity": "strong", "image_url": "/x", "model_aesthetic_score": model_scores[3]},
        ],
        "human_preferences": human_rows,
    }


def test_average_ranks_handles_ties():
    assert _average_ranks([5, 4, 4, 1]) == [1.0, 2.5, 2.5, 4.0]


def test_dedupe_preferences_keeps_latest_per_participant():
    rows = [
        {"participant_code": "P001", "value": 1},
        {"participant_code": "P002", "value": 2},
        {"participant_code": "P001", "value": 3},
    ]
    deduped = dedupe_preferences(rows)
    values = {row["participant_code"]: row["value"] for row in deduped}
    assert values == {"P001": 3, "P002": 2}


def test_perfect_order_agreement_has_positive_correlations_and_choice_agreement():
    result = analyze_batch_handoff(_handoff())
    assert result["participant_count"] == 3
    assert result["top_choice_agreement"] is True
    assert result["pearson_r"] > 0.9
    assert result["spearman_rho"] == 1.0
    assert result["human_consensus_candidate_id"] == "original:o1"


def test_top_choice_mismatch_is_reported():
    rows = [
        {
            "participant_code": "P001",
            "ratings": {"original:o1": 2, "cf-s": 3, "cf-m": 4, "cf-x": 5},
            "preferred_candidate_id": "cf-x",
        },
        {
            "participant_code": "P002",
            "ratings": {"original:o1": 2, "cf-s": 3, "cf-m": 4, "cf-x": 5},
            "preferred_candidate_id": "cf-x",
        },
        {
            "participant_code": "P003",
            "ratings": {"original:o1": 2, "cf-s": 3, "cf-m": 4, "cf-x": 5},
            "preferred_candidate_id": "cf-x",
        },
    ]
    result = analyze_batch_handoff(_handoff(human_rows=rows))
    assert result["top_choice_agreement"] is False
    codes = {issue["code"] for issue in result["issues"]}
    assert "TOP_CHOICE_MISMATCH" in codes
    assert result["spearman_rho"] == -1.0


def test_constant_human_ratings_make_correlation_unavailable():
    ratings = {"original:o1": 4, "cf-s": 4, "cf-m": 4, "cf-x": 4}
    rows = [
        {"participant_code": "P001", "ratings": ratings, "preferred_candidate_id": "original:o1"},
        {"participant_code": "P002", "ratings": ratings, "preferred_candidate_id": "original:o1"},
        {"participant_code": "P003", "ratings": ratings, "preferred_candidate_id": "original:o1"},
    ]
    result = analyze_batch_handoff(_handoff(human_rows=rows))
    assert result["pearson_r"] is None
    assert result["spearman_rho"] is None
    assert any(issue["code"] == "CORRELATION_UNAVAILABLE" for issue in result["issues"])


def test_dashboard_aggregates_batches():
    one = analyze_batch_handoff(_handoff())
    two = analyze_batch_handoff(_handoff())
    two["batch_id"] = "batch-2"
    two["region"] = {"label": "full_image", "is_fallback": True}
    dashboard = summarize_dashboard([one, two])
    assert dashboard["batch_count"] == 2
    assert dashboard["participant_response_count"] == 6
    assert dashboard["top_choice_agreement_rate"] == 1.0
    assert {row["name"] for row in dashboard["by_region_mode"]} == {"detected_region", "full_image"}


def test_pdf_aligned_metrics_include_mae_rmse_direction_and_slopes():
    result = analyze_batch_handoff(_handoff())
    assert result["mae"] is not None
    assert result["rmse"] is not None
    assert result["rmse"] >= result["mae"]
    assert result["directional_agreement_rate"] == 1.0
    assert result["disagreement_rate"] >= 0.0
    assert result["model_severity_response_slope"] < 0
    assert result["human_severity_response_slope"] < 0
    assert len(result["severity_response"]) == 4


def test_sensitivity_rows_include_effect_size_and_confidence_interval():
    result = analyze_batch_handoff(_handoff())
    assert len(result["perturbation_sensitivity"]) == 3
    for row in result["perturbation_sensitivity"]:
        assert "absolute_sensitivity_disagreement" in row
        assert "directional_agreement" in row
        assert row["human_delta_ci_95"] is not None
        assert row["human_effect_size_cohens_dz"] is not None


def test_human_preference_rate_is_derived_from_best_votes():
    result = analyze_batch_handoff(_handoff())
    rates = result["human_preference_rate"]
    assert rates["original_preferred_rate"] == 1.0
    assert rates["counterfactual_preferred_rate"] == 0.0
    assert rates["same_rate"] is None


def test_dashboard_contains_pdf_required_visualization_data():
    one = analyze_batch_handoff(_handoff())
    dashboard = summarize_dashboard([one])
    assert dashboard["mae"] is not None
    assert dashboard["rmse"] is not None
    assert dashboard["directional_agreement_rate"] == 1.0
    assert dashboard["top_choice_agreement_ci_95"] is not None
    assert len(dashboard["correlation_points"]) == 4
    assert dashboard["by_perturbation"]
    assert dashboard["by_region_label"]
    assert dashboard["by_severity"]
    assert dashboard["heatmap"]
    assert dashboard["findings"]


def test_dashboard_reports_project_preference_rates():
    dashboard = summarize_dashboard([analyze_batch_handoff(_handoff())])
    assert dashboard["original_preference_rate"] == 1.0
    assert dashboard["counterfactual_preference_rate"] == 0.0
    assert dashboard["by_perturbation"][0]["mean_model_sensitivity"] is not None
    assert dashboard["by_perturbation"][0]["mean_human_sensitivity"] is not None


def test_exact_pairwise_protocol_overrides_derived_best_vote_rates():
    handoff = _handoff()
    handoff["pairwise_preferences"] = [
        {"participant_code": "P001", "counterfactual_id": "cf-s", "severity": "subtle", "choice": "original"},
        {"participant_code": "P001", "counterfactual_id": "cf-m", "severity": "moderate", "choice": "same"},
        {"participant_code": "P001", "counterfactual_id": "cf-x", "severity": "strong", "choice": "modified"},
        {"participant_code": "P002", "counterfactual_id": "cf-s", "severity": "subtle", "choice": "original"},
        {"participant_code": "P002", "counterfactual_id": "cf-m", "severity": "moderate", "choice": "original"},
        {"participant_code": "P002", "counterfactual_id": "cf-x", "severity": "strong", "choice": "same"},
    ]
    result = analyze_batch_handoff(handoff)
    rates = result["human_preference_rate"]
    assert rates["original_preferred_rate"] == 0.5
    assert rates["counterfactual_preferred_rate"] == round(1 / 6, 4)
    assert rates["same_rate"] == round(2 / 6, 4)
    assert "Exact PDF pairwise protocol" in rates["protocol_note"]
    assert len(result["human_preference_rate_by_severity"]) == 3


def test_dashboard_aggregates_image_category():
    dashboard = summarize_dashboard([analyze_batch_handoff(_handoff())])
    assert dashboard["by_category"]
    assert dashboard["by_category"][0]["name"] == "realistic"
