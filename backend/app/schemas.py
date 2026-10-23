from typing import Any, Literal

from pydantic import BaseModel, Field

Severity = Literal["subtle", "moderate", "strong"]


class DatasetRecordOut(BaseModel):
    name: str
    prompt: str
    adj1: str | None = None
    adj2: str | None = None
    style: str | None = None
    mos_quality: float
    std_quality: float
    mos_align: float
    std_align: float
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

    mos_quality: float
    std_quality: float
    mos_align: float
    std_align: float

    width: int
    height: int

    original_image_url: str
    annotated_image_url: str | None = None

    context: dict[str, Any]
    detector: dict[str, Any]
    regions: list[RegionOut]
    created_at: str


class HandoffOut(BaseModel):
    schema_version: str = "1.4-core"
    image_id: str
    dataset_name: str = "AGIQA-3K"
    dataset_image_name: str
    original_image_url: str
    prompt: str
    generator: str | None = None

    agiqa3k_ground_truth: dict[str, Any]
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
    original_image_id: str
    original_image_url: str
    dataset_image_name: str
    prompt: str
    generator: str | None = None
    agiqa3k_ground_truth: dict[str, Any]
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
