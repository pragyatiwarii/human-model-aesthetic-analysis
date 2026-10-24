from __future__ import annotations

from collections import Counter
from math import sqrt
from statistics import mean, pstdev, stdev
from typing import Any


SEVERITY_ORDER = {None: 0, "subtle": 1, "moderate": 2, "strong": 3}
DISAGREEMENT_SCORE_GAP_THRESHOLD = 0.75
DIRECTION_EPSILON = 0.05

# Two-sided 95% Student-t critical values (0.975 quantile) for df 1..30.
_T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def dedupe_preferences(preferences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the latest response for each participant code within one batch."""
    latest: dict[str, dict[str, Any]] = {}
    for row in preferences:
        code = str(row.get("participant_code", "")).strip()
        if not code:
            continue
        latest[code] = row
    return list(latest.values())


def _average_ranks(values: list[float], *, descending: bool = True) -> list[float]:
    indexed = list(enumerate(values))
    indexed.sort(key=lambda item: item[1], reverse=descending)
    ranks = [0.0] * len(values)
    position = 1
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        group_positions = list(range(position, position + (end - cursor)))
        average_rank = sum(group_positions) / len(group_positions)
        for idx, _value in indexed[cursor:end]:
            ranks[idx] = average_rank
        position += end - cursor
        cursor = end
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    x_dev = [value - x_mean for value in xs]
    y_dev = [value - y_mean for value in ys]
    denom = sqrt(sum(value * value for value in x_dev) * sum(value * value for value in y_dev))
    if denom == 0:
        return None
    return sum(a * b for a, b in zip(x_dev, y_dev)) / denom


def _linear_slope(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_bar = mean(xs)
    y_bar = mean(ys)
    denom = sum((x - x_bar) ** 2 for x in xs)
    if denom == 0:
        return None
    return sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys)) / denom


def _mae(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or not xs:
        return None
    return mean(abs(x - y) for x, y in zip(xs, ys))


def _rmse(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or not xs:
        return None
    return sqrt(mean((x - y) ** 2 for x, y in zip(xs, ys)))


def _t_critical_95(df: int) -> float:
    if df <= 0:
        return 1.96
    if df <= 30:
        return _T975[df]
    if df <= 40:
        return 2.021
    if df <= 60:
        return 2.000
    if df <= 120:
        return 1.980
    return 1.960


def _mean_ci_95(values: list[float]) -> dict[str, float] | None:
    if len(values) < 2:
        return None
    center = mean(values)
    sd = stdev(values)
    margin = _t_critical_95(len(values) - 1) * sd / sqrt(len(values))
    return {
        "mean": round(center, 4),
        "lower": round(center - margin, 4),
        "upper": round(center + margin, 4),
        "n": len(values),
    }


def _wilson_ci_95(successes: int, total: int) -> dict[str, float] | None:
    if total <= 0:
        return None
    z = 1.96
    p = successes / total
    denom = 1 + (z * z / total)
    center = (p + (z * z / (2 * total))) / denom
    half = z * sqrt((p * (1 - p) / total) + (z * z / (4 * total * total))) / denom
    return {
        "estimate": round(p, 4),
        "lower": round(max(0.0, center - half), 4),
        "upper": round(min(1.0, center + half), 4),
        "n": total,
    }


def _cohens_dz(differences: list[float]) -> float | None:
    """Paired-samples Cohen's dz for counterfactual - original human ratings."""
    if len(differences) < 2:
        return None
    sd = stdev(differences)
    if sd == 0:
        return None
    return mean(differences) / sd


def _direction(value: float, epsilon: float = DIRECTION_EPSILON) -> int:
    if value > epsilon:
        return 1
    if value < -epsilon:
        return -1
    return 0


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def _candidate_label(candidate: dict[str, Any]) -> str:
    if candidate.get("source_type") == "original":
        return "Original"
    severity = candidate.get("severity") or "counterfactual"
    return str(severity).replace("_", " ").title()


def analyze_batch_handoff(handoff: dict[str, Any]) -> dict[str, Any]:
    """Compute PDF-aligned human-model quantitative metrics for one Module 3 batch."""
    candidates = handoff.get("candidates") or []
    preferences = dedupe_preferences(handoff.get("human_preferences") or [])
    if not candidates:
        raise ValueError("Module 4 requires candidate records from Module 3.")
    if not preferences:
        raise ValueError("Module 4 requires at least one human preference response.")

    candidate_ids = [candidate["candidate_id"] for candidate in candidates]
    candidate_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}

    valid_preferences: list[dict[str, Any]] = []
    for pref in preferences:
        ratings = pref.get("ratings") or {}
        if all(candidate_id in ratings for candidate_id in candidate_ids):
            valid_preferences.append(pref)
    preferences = valid_preferences
    if not preferences:
        raise ValueError("No complete human preference response is available for this batch.")

    human_means: dict[str, float] = {}
    human_stds: dict[str, float] = {}
    human_cis: dict[str, dict[str, float] | None] = {}
    preference_counts = Counter(pref["preferred_candidate_id"] for pref in preferences)

    for candidate_id in candidate_ids:
        ratings = [float(pref["ratings"][candidate_id]) for pref in preferences]
        human_means[candidate_id] = mean(ratings)
        human_stds[candidate_id] = pstdev(ratings) if len(ratings) > 1 else 0.0
        human_cis[candidate_id] = _mean_ci_95(ratings)

    model_scores = [float(candidate_by_id[candidate_id]["model_aesthetic_score"]) for candidate_id in candidate_ids]
    human_scores = [human_means[candidate_id] for candidate_id in candidate_ids]
    model_ranks = _average_ranks(model_scores, descending=True)
    human_ranks = _average_ranks(human_scores, descending=True)

    pearson = _pearson(model_scores, human_scores)
    spearman = _pearson(model_ranks, human_ranks)
    mae = _mae(model_scores, human_scores)
    rmse = _rmse(model_scores, human_scores)

    model_preferred = max(
        candidate_ids,
        key=lambda cid: (
            float(candidate_by_id[cid]["model_aesthetic_score"]),
            -SEVERITY_ORDER.get(candidate_by_id[cid].get("severity"), 9),
        ),
    )

    consensus_sorted = sorted(
        candidate_ids,
        key=lambda cid: (
            preference_counts.get(cid, 0),
            human_means[cid],
            -SEVERITY_ORDER.get(candidate_by_id[cid].get("severity"), 9),
        ),
        reverse=True,
    )
    human_consensus = consensus_sorted[0]
    top_votes = preference_counts.get(human_consensus, 0)
    top_ties = [cid for cid in candidate_ids if preference_counts.get(cid, 0) == top_votes]
    consensus_strength = top_votes / len(preferences)

    original_id = next(
        (candidate["candidate_id"] for candidate in candidates if candidate.get("source_type") == "original"),
        None,
    )
    if original_id is None:
        raise ValueError("Module 4 requires the original candidate to compute deltas and sensitivity.")

    original_model = float(candidate_by_id[original_id]["model_aesthetic_score"])
    original_human = human_means[original_id]

    candidate_rows: list[dict[str, Any]] = []
    for index, candidate_id in enumerate(candidate_ids):
        candidate = candidate_by_id[candidate_id]
        model_score = float(candidate["model_aesthetic_score"])
        human_mean = human_means[candidate_id]
        score_gap = abs(model_score - human_mean)
        rank_gap = abs(model_ranks[index] - human_ranks[index])
        candidate_rows.append(
            {
                "candidate_id": candidate_id,
                "label": _candidate_label(candidate),
                "source_type": candidate.get("source_type"),
                "severity": candidate.get("severity"),
                "image_url": candidate.get("image_url"),
                "model_score": round(model_score, 4),
                "human_mean_rating": round(human_mean, 4),
                "human_rating_std": round(human_stds[candidate_id], 4),
                "human_mean_ci_95": human_cis[candidate_id],
                "rating_count": len(preferences),
                "preference_count": int(preference_counts.get(candidate_id, 0)),
                "preference_share": round(preference_counts.get(candidate_id, 0) / len(preferences), 4),
                "model_rank": round(model_ranks[index], 3),
                "human_rank": round(human_ranks[index], 3),
                "absolute_score_gap": round(score_gap, 4),
                "absolute_rank_gap": round(rank_gap, 3),
                "substantial_disagreement": score_gap >= DISAGREEMENT_SCORE_GAP_THRESHOLD,
                "model_delta_from_original": round(model_score - original_model, 4),
                "human_delta_from_original": round(human_mean - original_human, 4),
            }
        )

    mean_abs_gap = mean(row["absolute_score_gap"] for row in candidate_rows)
    mean_rank_gap = mean(row["absolute_rank_gap"] for row in candidate_rows)
    strongest = max(candidate_rows, key=lambda row: (row["absolute_score_gap"], row["absolute_rank_gap"]))
    substantial_count = sum(1 for row in candidate_rows if row["substantial_disagreement"])
    disagreement_rate = substantial_count / len(candidate_rows)

    # PDF-aligned perturbation sensitivity: human/model response and absolute disagreement.
    sensitivity: list[dict[str, Any]] = []
    for row in sorted(candidate_rows, key=lambda item: SEVERITY_ORDER.get(item.get("severity"), 0)):
        if row["source_type"] != "counterfactual":
            continue
        candidate_id = row["candidate_id"]
        paired_differences = [
            float(pref["ratings"][candidate_id]) - float(pref["ratings"][original_id])
            for pref in preferences
        ]
        human_delta = mean(paired_differences)
        model_delta = float(row["model_delta_from_original"])
        human_direction = _direction(human_delta)
        model_direction = _direction(model_delta)
        sensitivity.append(
            {
                "candidate_id": candidate_id,
                "severity": row["severity"],
                "severity_index": SEVERITY_ORDER.get(row.get("severity"), 0),
                "model_delta_from_original": round(model_delta, 4),
                "human_delta_from_original": round(human_delta, 4),
                "model_sensitivity": round(abs(model_delta), 4),
                "human_sensitivity": round(abs(human_delta), 4),
                "absolute_sensitivity_disagreement": round(abs(model_delta - human_delta), 4),
                "directional_agreement": human_direction == model_direction,
                "model_direction": model_direction,
                "human_direction": human_direction,
                "human_delta_ci_95": _mean_ci_95(paired_differences),
                "human_effect_size_cohens_dz": _round_or_none(_cohens_dz(paired_differences)),
            }
        )

    directional_agreement_rate = (
        sum(1 for row in sensitivity if row["directional_agreement"]) / len(sensitivity)
        if sensitivity else None
    )

    ordered_candidates = sorted(candidate_rows, key=lambda row: SEVERITY_ORDER.get(row.get("severity"), 0))
    severity_x = [float(SEVERITY_ORDER.get(row.get("severity"), 0)) for row in ordered_candidates]
    severity_model_y = [float(row["model_score"]) for row in ordered_candidates]
    severity_human_y = [float(row["human_mean_rating"]) for row in ordered_candidates]
    model_slope = _linear_slope(severity_x, severity_model_y)
    human_slope = _linear_slope(severity_x, severity_human_y)

    severity_response = [
        {
            "severity": row["severity"] or "original",
            "severity_index": SEVERITY_ORDER.get(row.get("severity"), 0),
            "model_score": row["model_score"],
            "human_mean_rating": row["human_mean_rating"],
            "human_mean_ci_95": row["human_mean_ci_95"],
        }
        for row in ordered_candidates
    ]

    # Prefer the exact PDF protocol (Original / Modified / About the same) when
    # pairwise annotations are available. Otherwise retain backward-compatible
    # Best-of-four derived rates for older batches.
    pairwise = handoff.get("pairwise_preferences") or []
    pairwise_counts = Counter(row.get("choice") for row in pairwise)
    human_preference_rate_by_severity: list[dict[str, Any]] = []
    if pairwise:
        total_pairwise = len(pairwise)
        human_preference_rate = {
            "original_preferred_rate": round(pairwise_counts.get("original", 0) / total_pairwise, 4),
            "counterfactual_preferred_rate": round(pairwise_counts.get("modified", 0) / total_pairwise, 4),
            "same_rate": round(pairwise_counts.get("same", 0) / total_pairwise, 4),
            "protocol_note": "Exact PDF pairwise protocol: Original / Modified / About the same.",
        }
        for severity in ("subtle", "moderate", "strong"):
            rows = [row for row in pairwise if row.get("severity") == severity]
            if not rows:
                continue
            counts = Counter(row.get("choice") for row in rows)
            human_preference_rate_by_severity.append({
                "severity": severity,
                "response_count": len(rows),
                "original_rate": round(counts.get("original", 0) / len(rows), 4),
                "modified_rate": round(counts.get("modified", 0) / len(rows), 4),
                "same_rate": round(counts.get("same", 0) / len(rows), 4),
            })
    else:
        original_preference_count = int(preference_counts.get(original_id, 0))
        modified_preference_count = len(preferences) - original_preference_count
        human_preference_rate = {
            "original_preferred_rate": round(original_preference_count / len(preferences), 4),
            "counterfactual_preferred_rate": round(modified_preference_count / len(preferences), 4),
            "same_rate": None,
            "protocol_note": "Backward-compatible derived rate from blind Best-of-four; collect pairwise responses for exact PDF compliance.",
        }

    issues: list[dict[str, Any]] = []
    if len(preferences) < 3:
        issues.append(
            {
                "code": "LOW_PARTICIPANT_COUNT",
                "level": "warning",
                "message": "Fewer than 3 unique human participants are available; treat agreement metrics as a software/demo result, not a study conclusion.",
            }
        )
    if len(top_ties) > 1:
        issues.append(
            {
                "code": "HUMAN_CONSENSUS_TIE",
                "level": "warning",
                "message": "Human Best votes are tied; the displayed consensus uses mean human rating as a tie-breaker.",
            }
        )
    elif consensus_strength < 0.5:
        issues.append(
            {
                "code": "LOW_HUMAN_CONSENSUS",
                "level": "warning",
                "message": f"Only {consensus_strength:.0%} of participants selected the consensus candidate as Best.",
            }
        )
    if model_preferred != human_consensus:
        issues.append(
            {
                "code": "TOP_CHOICE_MISMATCH",
                "level": "error",
                "message": "The model-preferred candidate differs from the human-consensus preferred candidate.",
                "model_candidate_id": model_preferred,
                "human_candidate_id": human_consensus,
            }
        )
    for row in [r for r in candidate_rows if r["substantial_disagreement"]]:
        issues.append(
            {
                "code": "HIGH_SCORE_GAP",
                "level": "error",
                "candidate_id": row["candidate_id"],
                "message": f'{row["label"]} has a human–model score gap of {row["absolute_score_gap"]:.2f} on the 1–5 scale.',
            }
        )
    for row in [r for r in candidate_rows if r["absolute_rank_gap"] >= 2]:
        issues.append(
            {
                "code": "RANK_REVERSAL",
                "level": "warning",
                "candidate_id": row["candidate_id"],
                "message": f'{row["label"]} differs by {row["absolute_rank_gap"]:.1f} rank positions between humans and the model.',
            }
        )
    if pearson is None or spearman is None:
        issues.append(
            {
                "code": "CORRELATION_UNAVAILABLE",
                "level": "info",
                "message": "Correlation is unavailable because one side has no variation across the four candidates.",
            }
        )
    elif pearson < 0:
        issues.append(
            {
                "code": "NEGATIVE_LINEAR_AGREEMENT",
                "level": "error",
                "message": f"Pearson correlation is negative ({pearson:.3f}), indicating opposing score movement in this batch.",
            }
        )
    if directional_agreement_rate is not None and directional_agreement_rate < 0.5:
        issues.append(
            {
                "code": "LOW_DIRECTIONAL_AGREEMENT",
                "level": "error",
                "message": f"Human and model response directions agree for only {directional_agreement_rate:.0%} of severity levels.",
            }
        )

    issue_counts = Counter(issue["level"] for issue in issues)
    return {
        "schema_version": "4.1-pdf-aligned-analysis",
        "batch_id": handoff["batch_id"],
        "original": handoff.get("original") or {},
        "region": handoff.get("region") or {},
        "perturbation": handoff.get("perturbation") or {},
        "participant_count": len(preferences),
        "model_preferred_candidate_id": model_preferred,
        "human_consensus_candidate_id": human_consensus,
        "top_choice_agreement": model_preferred == human_consensus,
        "human_consensus_strength": round(consensus_strength, 4),
        "pearson_r": _round_or_none(pearson),
        "spearman_rho": _round_or_none(spearman),
        "mae": _round_or_none(mae),
        "rmse": _round_or_none(rmse),
        "mean_absolute_score_gap": round(mean_abs_gap, 4),
        "mean_absolute_rank_gap": round(mean_rank_gap, 4),
        "disagreement_score_gap_threshold": DISAGREEMENT_SCORE_GAP_THRESHOLD,
        "disagreement_rate": round(disagreement_rate, 4),
        "directional_agreement_rate": _round_or_none(directional_agreement_rate),
        "model_severity_response_slope": _round_or_none(model_slope),
        "human_severity_response_slope": _round_or_none(human_slope),
        "human_preference_rate": human_preference_rate,
        "human_preference_rate_by_severity": human_preference_rate_by_severity,
        "strongest_disagreement_candidate_id": strongest["candidate_id"],
        "candidates": candidate_rows,
        "perturbation_sensitivity": sensitivity,
        "severity_response": severity_response,
        "issues": issues,
        "issue_counts": {
            "error": int(issue_counts.get("error", 0)),
            "warning": int(issue_counts.get("warning", 0)),
            "info": int(issue_counts.get("info", 0)),
        },
    }


def _group_aggregate(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    candidate_rows = [candidate for row in rows for candidate in row["candidates"]]
    sensitivity_rows = [item for row in rows for item in row["perturbation_sensitivity"]]
    directional = [item for item in sensitivity_rows if item.get("directional_agreement") is not None]
    return {
        "name": name,
        "batch_count": len(rows),
        "participant_response_count": sum(row["participant_count"] for row in rows),
        "top_choice_agreement_rate": round(sum(1 for row in rows if row["top_choice_agreement"]) / len(rows), 4),
        "disagreement_rate": round(sum(1 for candidate in candidate_rows if candidate["substantial_disagreement"]) / len(candidate_rows), 4) if candidate_rows else None,
        "mean_absolute_score_gap": round(mean(candidate["absolute_score_gap"] for candidate in candidate_rows), 4) if candidate_rows else None,
        "mean_model_delta": round(mean(item["model_delta_from_original"] for item in sensitivity_rows), 4) if sensitivity_rows else None,
        "mean_human_delta": round(mean(item["human_delta_from_original"] for item in sensitivity_rows), 4) if sensitivity_rows else None,
        "mean_model_sensitivity": round(mean(item["model_sensitivity"] for item in sensitivity_rows), 4) if sensitivity_rows else None,
        "mean_human_sensitivity": round(mean(item["human_sensitivity"] for item in sensitivity_rows), 4) if sensitivity_rows else None,
        "mean_sensitivity_disagreement": round(mean(item["absolute_sensitivity_disagreement"] for item in sensitivity_rows), 4) if sensitivity_rows else None,
        "directional_agreement_rate": round(sum(1 for item in directional if item["directional_agreement"]) / len(directional), 4) if directional else None,
        "mean_model_severity_slope": _round_or_none(mean([row["model_severity_response_slope"] for row in rows if row.get("model_severity_response_slope") is not None])) if any(row.get("model_severity_response_slope") is not None for row in rows) else None,
        "mean_human_severity_slope": _round_or_none(mean([row["human_severity_response_slope"] for row in rows if row.get("human_severity_response_slope") is not None])) if any(row.get("human_severity_response_slope") is not None for row in rows) else None,
    }


def summarize_dashboard(batch_analyses: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch_analyses:
        return {
            "schema_version": "4.1-pdf-aligned-dashboard",
            "batch_count": 0,
            "participant_response_count": 0,
            "top_choice_agreement_rate": None,
            "top_choice_agreement_ci_95": None,
            "directional_agreement_rate": None,
            "disagreement_rate": None,
            "original_preference_rate": None,
            "counterfactual_preference_rate": None,
            "mean_pearson_r": None,
            "mean_spearman_rho": None,
            "mae": None,
            "rmse": None,
            "mean_absolute_score_gap": None,
            "mean_absolute_score_gap_ci_95": None,
            "by_perturbation": [],
            "by_region_mode": [],
            "by_region_label": [],
            "by_category": [],
            "by_severity": [],
            "correlation_points": [],
            "heatmap": [],
            "effect_size_summary": [],
            "top_disagreement_batches": [],
            "findings": [],
            "batches": [],
            "total_error_flags": 0,
            "total_warning_flags": 0,
        }

    pearsons = [row["pearson_r"] for row in batch_analyses if row.get("pearson_r") is not None]
    spearmans = [row["spearman_rho"] for row in batch_analyses if row.get("spearman_rho") is not None]
    candidate_rows = [candidate for row in batch_analyses for candidate in row["candidates"]]
    sensitivity_rows = [
        {**item, "batch_id": row["batch_id"], "perturbation": (row.get("perturbation") or {}).get("label") or (row.get("perturbation") or {}).get("type") or "Unknown", "region_label": (row.get("region") or {}).get("label") or "Unknown"}
        for row in batch_analyses for item in row["perturbation_sensitivity"]
    ]

    top_choice_successes = sum(1 for row in batch_analyses if row["top_choice_agreement"])
    total_preference_responses = sum(row["participant_count"] for row in batch_analyses)
    original_preference_votes = sum(
        row["human_preference_rate"]["original_preferred_rate"] * row["participant_count"]
        for row in batch_analyses
    )
    all_directional = [item for item in sensitivity_rows if item.get("directional_agreement") is not None]
    substantial_candidates = [candidate for candidate in candidate_rows if candidate["substantial_disagreement"]]
    model_all = [candidate["model_score"] for candidate in candidate_rows]
    human_all = [candidate["human_mean_rating"] for candidate in candidate_rows]

    def aggregate_groups(key_fn):
        groups: dict[str, list[dict[str, Any]]] = {}
        for analysis in batch_analyses:
            groups.setdefault(key_fn(analysis), []).append(analysis)
        return [_group_aggregate(rows, name) for name, rows in sorted(groups.items())]

    by_perturbation = aggregate_groups(lambda row: str((row.get("perturbation") or {}).get("label") or (row.get("perturbation") or {}).get("type") or "Unknown"))
    by_region_mode = aggregate_groups(lambda row: "full_image" if (row.get("region") or {}).get("is_fallback") else "detected_region")
    by_region_label = aggregate_groups(lambda row: str((row.get("region") or {}).get("label") or "Unknown"))
    by_category = aggregate_groups(lambda row: str((row.get("original") or {}).get("category") or (row.get("original") or {}).get("generator") or "Unknown"))

    by_severity: list[dict[str, Any]] = []
    for severity in ("subtle", "moderate", "strong"):
        rows = [item for item in sensitivity_rows if item.get("severity") == severity]
        if not rows:
            continue
        effects = [item["human_effect_size_cohens_dz"] for item in rows if item.get("human_effect_size_cohens_dz") is not None]
        by_severity.append(
            {
                "severity": severity,
                "severity_index": SEVERITY_ORDER[severity],
                "observation_count": len(rows),
                "mean_model_delta": round(mean(item["model_delta_from_original"] for item in rows), 4),
                "mean_human_delta": round(mean(item["human_delta_from_original"] for item in rows), 4),
                "mean_model_sensitivity": round(mean(item["model_sensitivity"] for item in rows), 4),
                "mean_human_sensitivity": round(mean(item["human_sensitivity"] for item in rows), 4),
                "mean_absolute_sensitivity_disagreement": round(mean(item["absolute_sensitivity_disagreement"] for item in rows), 4),
                "directional_agreement_rate": round(sum(1 for item in rows if item["directional_agreement"]) / len(rows), 4),
                "mean_human_effect_size_cohens_dz": _round_or_none(mean(effects) if effects else None),
            }
        )

    # Region × perturbation heatmap; value is candidate-level substantial disagreement rate.
    heatmap_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for analysis in batch_analyses:
        region_name = str((analysis.get("region") or {}).get("label") or "Unknown")
        perturbation_name = str((analysis.get("perturbation") or {}).get("label") or (analysis.get("perturbation") or {}).get("type") or "Unknown")
        heatmap_groups.setdefault((region_name, perturbation_name), []).append(analysis)
    heatmap = []
    for (region_name, perturbation_name), rows in sorted(heatmap_groups.items()):
        cells = [candidate for row in rows for candidate in row["candidates"]]
        heatmap.append(
            {
                "region": region_name,
                "perturbation": perturbation_name,
                "batch_count": len(rows),
                "disagreement_rate": round(sum(1 for candidate in cells if candidate["substantial_disagreement"]) / len(cells), 4) if cells else None,
                "mean_absolute_score_gap": round(mean(candidate["absolute_score_gap"] for candidate in cells), 4) if cells else None,
            }
        )

    correlation_points = []
    for analysis in batch_analyses:
        for candidate in analysis["candidates"]:
            correlation_points.append(
                {
                    "batch_id": analysis["batch_id"],
                    "candidate_id": candidate["candidate_id"],
                    "candidate": candidate["label"],
                    "severity": candidate.get("severity") or "original",
                    "model_score": candidate["model_score"],
                    "human_mean_rating": candidate["human_mean_rating"],
                    "perturbation": (analysis.get("perturbation") or {}).get("label") or (analysis.get("perturbation") or {}).get("type"),
                    "region": (analysis.get("region") or {}).get("label"),
                }
            )

    effect_groups: dict[tuple[str, str], list[float]] = {}
    for item in sensitivity_rows:
        effect = item.get("human_effect_size_cohens_dz")
        if effect is not None:
            effect_groups.setdefault((str(item["perturbation"]), str(item["severity"])), []).append(float(effect))
    effect_size_summary = [
        {
            "perturbation": perturbation,
            "severity": severity,
            "observation_count": len(values),
            "mean_cohens_dz": round(mean(values), 4),
        }
        for (perturbation, severity), values in sorted(effect_groups.items())
    ]

    batch_summaries = []
    for analysis in batch_analyses:
        batch_summaries.append(
            {
                "batch_id": analysis["batch_id"],
                "dataset_image_name": (analysis.get("original") or {}).get("dataset_image_name"),
                "prompt": (analysis.get("original") or {}).get("prompt"),
                "perturbation": (analysis.get("perturbation") or {}).get("label") or (analysis.get("perturbation") or {}).get("type"),
                "region_label": (analysis.get("region") or {}).get("label"),
                "participant_count": analysis["participant_count"],
                "top_choice_agreement": analysis["top_choice_agreement"],
                "directional_agreement_rate": analysis.get("directional_agreement_rate"),
                "disagreement_rate": analysis.get("disagreement_rate"),
                "pearson_r": analysis["pearson_r"],
                "spearman_rho": analysis["spearman_rho"],
                "mae": analysis.get("mae"),
                "rmse": analysis.get("rmse"),
                "mean_absolute_score_gap": analysis["mean_absolute_score_gap"],
                "model_severity_response_slope": analysis.get("model_severity_response_slope"),
                "human_severity_response_slope": analysis.get("human_severity_response_slope"),
                "error_count": analysis["issue_counts"]["error"],
                "warning_count": analysis["issue_counts"]["warning"],
            }
        )

    top_disagreements = sorted(
        batch_summaries,
        key=lambda row: (
            not row["top_choice_agreement"],
            row["error_count"],
            row["disagreement_rate"] or 0,
            row["mean_absolute_score_gap"],
        ),
        reverse=True,
    )[:10]

    findings: list[dict[str, str]] = []
    if by_perturbation:
        worst_pert = max(by_perturbation, key=lambda row: (row.get("disagreement_rate") or 0, row.get("mean_absolute_score_gap") or 0))
        findings.append({"type": "disagreement", "text": f"Highest perturbation disagreement: {worst_pert['name']} ({(worst_pert.get('disagreement_rate') or 0) * 100:.1f}% substantial candidate disagreements)."})
    if by_region_label:
        worst_region = max(by_region_label, key=lambda row: (row.get("disagreement_rate") or 0, row.get("mean_absolute_score_gap") or 0))
        findings.append({"type": "region", "text": f"Highest region disagreement: {worst_region['name']} ({(worst_region.get('disagreement_rate') or 0) * 100:.1f}%)."})
    if by_severity:
        worst_severity = max(by_severity, key=lambda row: row["mean_absolute_sensitivity_disagreement"])
        findings.append({"type": "severity", "text": f"Largest mean human–model sensitivity gap occurs at {worst_severity['severity']} severity ({worst_severity['mean_absolute_sensitivity_disagreement']:.3f})."})

    gap_values = [float(candidate["absolute_score_gap"]) for candidate in candidate_rows]
    return {
        "schema_version": "4.1-pdf-aligned-dashboard",
        "batch_count": len(batch_analyses),
        "participant_response_count": sum(row["participant_count"] for row in batch_analyses),
        "top_choice_agreement_rate": round(top_choice_successes / len(batch_analyses), 4),
        "top_choice_agreement_ci_95": _wilson_ci_95(top_choice_successes, len(batch_analyses)),
        "directional_agreement_rate": round(sum(1 for item in all_directional if item["directional_agreement"]) / len(all_directional), 4) if all_directional else None,
        "disagreement_rate": round(len(substantial_candidates) / len(candidate_rows), 4) if candidate_rows else None,
        "original_preference_rate": round(original_preference_votes / total_preference_responses, 4) if total_preference_responses else None,
        "counterfactual_preference_rate": round(1 - (original_preference_votes / total_preference_responses), 4) if total_preference_responses else None,
        "mean_pearson_r": _round_or_none(mean(pearsons) if pearsons else None),
        "mean_spearman_rho": _round_or_none(mean(spearmans) if spearmans else None),
        "mae": _round_or_none(_mae(model_all, human_all)),
        "rmse": _round_or_none(_rmse(model_all, human_all)),
        "mean_absolute_score_gap": round(mean(gap_values), 4) if gap_values else None,
        "mean_absolute_score_gap_ci_95": _mean_ci_95(gap_values),
        "total_error_flags": sum(row["issue_counts"]["error"] for row in batch_analyses),
        "total_warning_flags": sum(row["issue_counts"]["warning"] for row in batch_analyses),
        "by_perturbation": by_perturbation,
        "by_region_mode": by_region_mode,
        "by_region_label": by_region_label,
        "by_category": by_category,
        "by_severity": by_severity,
        "correlation_points": correlation_points,
        "heatmap": heatmap,
        "effect_size_summary": effect_size_summary,
        "top_disagreement_batches": top_disagreements,
        "findings": findings,
        "batches": batch_summaries,
    }
