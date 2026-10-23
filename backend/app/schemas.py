from typing import Any

from pydantic import BaseModel, Field


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
