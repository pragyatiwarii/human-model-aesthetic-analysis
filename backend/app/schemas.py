from typing import Any, Literal

from pydantic import BaseModel, Field

Severity = Literal["subtle", "moderate", "strong"]


class DatasetRecordOut(BaseModel):
    name: str
    prompt: str
    adj1: str | None = None
    adj2: str | None = None
    style: str | None = None
    mos_quality: float | None = None
    std_quality: float | None = None
    mos_align: float | None = None
    std_align: float | None = None
    generator: str
    image_exists: bool
    image_url: str | None = None


class DatasetListOut(BaseModel):
    dataset: str = "AGIQA-3K"
    total: int
    offset: int
    limit: int
    records: list[DatasetRecordOut]


class RegionOut(BaseModel):
    region_id: str
    label: str
    confidence: float
    bbox: list[float] = Field(
        description="Normalized [x1, y1, x2, y2] in the range 0..1"
    )
    polygon: list[list[float]] | None = None
    mask_url: str | None = None
    is_fallback: bool = False


class AnalysisOut(BaseModel):
    image_id: str
    dataset_name: str = "AGIQA-3K"
    dataset_image_name: str
    prompt: str
    generator: str | None = None
    adj1: str | None = None
    adj2: str | None = None
    style: str | None = None

    mos_quality: float | None = None
    std_quality: float | None = None
    mos_align: float | None = None
    std_align: float | None = None

    width: int
    height: int

    original_image_url: str
    annotated_image_url: str | None = None

    context: dict[str, Any]
    detector: dict[str, Any]
    regions: list[RegionOut]
    created_at: str


class HandoffOut(BaseModel):
    schema_version: str = "1.5-context"
    image_id: str
    dataset_name: str = "AGIQA-3K"
    dataset_image_name: str
    original_image_url: str
    prompt: str
    generator: str | None = None

    agiqa3k_ground_truth: dict[str, Any]
    context: dict[str, Any]
    regions: list[RegionOut]


class PerturbationSpecOut(BaseModel):
    key: str
    label: str
    description: str
    region_controlled: bool = True
    severities: dict[str, dict[str, Any]]


class CounterfactualGenerateIn(BaseModel):
    image_id: str
    region_id: str
    perturbation_type: str
    severities: list[Severity] = Field(
        default_factory=lambda: ["subtle", "moderate", "strong"],
        min_length=1,
    )


class CounterfactualVariantOut(BaseModel):
    counterfactual_id: str
    batch_id: str
    image_id: str
    region_id: str
    region_label: str
    perturbation_type: str
    severity: Severity
    parameters: dict[str, Any]
    counterfactual_url: str
    validation: dict[str, Any]


class CounterfactualBatchOut(BaseModel):
    schema_version: str = "2.0-counterfactual"
    batch_id: str
    dataset_name: str = "AGIQA-3K"
    original_image_id: str
    original_image_url: str
    dataset_image_name: str
    prompt: str
    generator: str | None = None
    agiqa3k_ground_truth: dict[str, Any]
    context: dict[str, Any]
    region: RegionOut
    perturbation_type: str
    variants: list[CounterfactualVariantOut]


class Module3HandoffOut(BaseModel):
    schema_version: str = "2.0-module3-handoff"
    batch_id: str
    original: dict[str, Any]
    region: dict[str, Any]
    perturbation: dict[str, Any]
    counterfactuals: list[dict[str, Any]]


class AestheticModelStatusOut(BaseModel):
    trained: bool
    version: str
    model_path: str
    training_limit: int
    training_samples: int | None = None
    validation_samples: int | None = None
    source_images_used: int | None = None
    validation_metrics: dict[str, float] | None = None
    target: str | None = None
    notes: str | None = None


class AestheticTrainIn(BaseModel):
    force: bool = False


class AestheticScoreOut(BaseModel):
    score_id: str
    batch_id: str
    candidate_id: str
    source_type: Literal["original", "counterfactual"]
    severity: Severity | None = None
    image_url: str
    score: float
    model_version: str
    features: dict[str, float]


class AestheticBatchScoresOut(BaseModel):
    schema_version: str = "3.0-aesthetic-scores"
    batch_id: str
    model: dict[str, Any]
    candidates: list[AestheticScoreOut]
    model_preferred_candidate_id: str


class HumanPreferenceIn(BaseModel):
    batch_id: str
    participant_code: str = Field(min_length=1, max_length=50)
    ratings: dict[str, int]
    preferred_candidate_id: str
    notes: str | None = Field(default=None, max_length=300)


class HumanPreferenceOut(BaseModel):
    preference_id: str
    batch_id: str
    participant_code: str
    ratings: dict[str, int]
    preferred_candidate_id: str
    notes: str | None = None
    created_at: str | None = None


class PairwisePreferenceIn(BaseModel):
    batch_id: str
    participant_code: str = Field(min_length=1, max_length=50)
    responses: dict[str, Literal["original", "modified", "same"]]


class PairwisePreferenceOut(BaseModel):
    preference_id: str
    batch_id: str
    participant_code: str
    counterfactual_id: str
    severity: Severity
    choice: Literal["original", "modified", "same"]
    created_at: str | None = None


class PairwisePreferenceSubmissionOut(BaseModel):
    batch_id: str
    participant_code: str
    responses: list[PairwisePreferenceOut]


class Module4HandoffOut(BaseModel):
    schema_version: str = "3.0-module4-handoff"
    batch_id: str
    original: dict[str, Any]
    region: dict[str, Any]
    perturbation: dict[str, Any]
    model: dict[str, Any]
    candidates: list[dict[str, Any]]
    human_preferences: list[dict[str, Any]]
    pairwise_preferences: list[dict[str, Any]] = Field(default_factory=list)
    human_summary: dict[str, Any]


class AgreementCandidateOut(BaseModel):
    candidate_id: str
    label: str
    source_type: str
    severity: str | None = None
    image_url: str | None = None
    model_score: float
    human_mean_rating: float
    human_rating_std: float
    human_mean_ci_95: dict[str, Any] | None = None
    rating_count: int
    preference_count: int
    preference_share: float
    model_rank: float
    human_rank: float
    absolute_score_gap: float
    absolute_rank_gap: float
    substantial_disagreement: bool = False
    model_delta_from_original: float | None = None
    human_delta_from_original: float | None = None


class AgreementIssueOut(BaseModel):
    code: str
    level: Literal["info", "warning", "error"]
    message: str
    candidate_id: str | None = None
    model_candidate_id: str | None = None
    human_candidate_id: str | None = None


class BatchAgreementOut(BaseModel):
    schema_version: str = "4.1-pdf-aligned-analysis"
    batch_id: str
    original: dict[str, Any]
    region: dict[str, Any]
    perturbation: dict[str, Any]
    participant_count: int
    model_preferred_candidate_id: str
    human_consensus_candidate_id: str
    top_choice_agreement: bool
    human_consensus_strength: float
    pearson_r: float | None = None
    spearman_rho: float | None = None
    mae: float | None = None
    rmse: float | None = None
    mean_absolute_score_gap: float
    mean_absolute_rank_gap: float
    disagreement_score_gap_threshold: float
    disagreement_rate: float
    directional_agreement_rate: float | None = None
    model_severity_response_slope: float | None = None
    human_severity_response_slope: float | None = None
    human_preference_rate: dict[str, Any]
    human_preference_rate_by_severity: list[dict[str, Any]] = Field(default_factory=list)
    strongest_disagreement_candidate_id: str
    candidates: list[AgreementCandidateOut]
    perturbation_sensitivity: list[dict[str, Any]]
    severity_response: list[dict[str, Any]]
    issues: list[AgreementIssueOut]
    issue_counts: dict[str, int]


class Module4DashboardOut(BaseModel):
    schema_version: str = "4.1-pdf-aligned-dashboard"
    batch_count: int
    participant_response_count: int
    top_choice_agreement_rate: float | None = None
    top_choice_agreement_ci_95: dict[str, Any] | None = None
    directional_agreement_rate: float | None = None
    disagreement_rate: float | None = None
    original_preference_rate: float | None = None
    counterfactual_preference_rate: float | None = None
    mean_pearson_r: float | None = None
    mean_spearman_rho: float | None = None
    mae: float | None = None
    rmse: float | None = None
    mean_absolute_score_gap: float | None = None
    mean_absolute_score_gap_ci_95: dict[str, Any] | None = None
    total_error_flags: int = 0
    total_warning_flags: int = 0
    by_perturbation: list[dict[str, Any]]
    by_region_mode: list[dict[str, Any]]
    by_region_label: list[dict[str, Any]]
    by_category: list[dict[str, Any]]
    by_severity: list[dict[str, Any]]
    correlation_points: list[dict[str, Any]]
    heatmap: list[dict[str, Any]]
    effect_size_summary: list[dict[str, Any]]
    top_disagreement_batches: list[dict[str, Any]]
    findings: list[dict[str, Any]]
    batches: list[dict[str, Any]]
