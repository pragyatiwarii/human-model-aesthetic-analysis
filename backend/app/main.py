from __future__ import annotations

import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.core.config import DATA_DIR, settings
from app.db import get_analysis, init_db, save_analysis
from app.schemas import (
    AnalysisOut,
    DatasetListOut,
    DatasetRecordOut,
    HandoffOut,
    RegionOut,
)
from app.services.agiqa3k import get_record, list_records, status
from app.services.context import build_context
from app.services.image_utils import create_annotated_preview
from app.services.region_detector import RegionDetector


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title=settings.app_name,
    version="1.4.0-module-1",
    description=(
        "AGIQA-3K input, reliable region detection/segmentation, "
        "and Module 2 handoff."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.frontend_origin,
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/media",
    StaticFiles(directory=DATA_DIR),
    name="media",
)

app.mount(
    "/dataset-images",
    StaticFiles(directory=settings.agiqa_images_dir),
    name="dataset-images",
)

detector = RegionDetector()


def region_out(image_id: str, region: dict) -> RegionOut:
    return RegionOut(
        region_id=region["region_id"],
        label=region["label"],
        confidence=region["confidence"],
        bbox=region["bbox"],
        polygon=region.get("polygon"),
        mask_url=(
            f'/media/runtime/regions/{image_id}/{region["mask_filename"]}'
            if region.get("mask_filename")
            else None
        ),
        is_fallback=bool(region.get("is_fallback")),
    )


def analysis_out(record: dict) -> AnalysisOut:
    return AnalysisOut(
        image_id=record["image_id"],
        dataset_image_name=record["dataset_image_name"],
        prompt=record["prompt"],
        generator=record["generator"],
        adj1=record["adj1"],
        adj2=record["adj2"],
        style=record["style"],
        mos_quality=record["mos_quality"],
        std_quality=record["std_quality"],
        mos_align=record["mos_align"],
        std_align=record["std_align"],
        width=record["width"],
        height=record["height"],
        original_image_url=(
            f'/media/runtime/uploads/{record["stored_filename"]}'
        ),
        annotated_image_url=(
            f'/media/runtime/annotated/{record["annotated_filename"]}'
            if record["annotated_filename"]
            else None
        ),
        context=record["context"],
        detector=record["detector"],
        regions=[
            region_out(record["image_id"], region)
            for region in record["regions"]
        ],
        created_at=record["created_at"],
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "module": 1,
        "owner": "Vennela Laxmi Puli",
        "dataset": "AGIQA-3K",
        "caption_model": "not used",
        "detection_confidence": settings.detection_confidence,
        "ml_models_enabled": settings.enable_ml_models,
    }


@app.get(f"{settings.api_prefix}/dataset/status")
def dataset_status():
    return status()


@app.get(
    f"{settings.api_prefix}/dataset/records",
    response_model=DatasetListOut,
)
def browse_records(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=25, ge=1, le=100),
    search: str | None = None,
    generator: str | None = None,
):
    records, total = list_records(
        offset=offset,
        limit=limit,
        search=search,
        generator=generator,
    )

    return DatasetListOut(
        total=total,
        offset=offset,
        limit=limit,
        records=[
            DatasetRecordOut(
                **record.to_dict(),
                image_url=(
                    f"/dataset-images/{record.name}"
                    if record.image_exists
                    else None
                ),
            )
            for record in records
        ],
    )


@app.get(
    f"{settings.api_prefix}/dataset/records/{{image_name}}",
    response_model=DatasetRecordOut,
)
def retrieve_dataset_record(image_name: str):
    record = get_record(image_name)

    if not record:
        raise HTTPException(
            status_code=404,
            detail="AGIQA-3K record not found.",
        )

    return DatasetRecordOut(
        **record.to_dict(),
        image_url=(
            f"/dataset-images/{record.name}"
            if record.image_exists
            else None
        ),
    )


@app.post(
    f"{settings.api_prefix}/dataset/analyze/{{image_name}}",
    response_model=AnalysisOut,
)
def analyze_dataset_image(image_name: str):
    record = get_record(image_name)

    if not record:
        raise HTTPException(
            status_code=404,
            detail="AGIQA-3K record not found.",
        )

    source_path = settings.agiqa_images_dir / record.name

    if not source_path.exists():
        raise HTTPException(
            status_code=409,
            detail=(
                "The metadata record exists, but the image file is missing. "
                "Copy the AGIQA-3K image archive into "
                "backend/data/agiqa3k/images/."
            ),
        )

    image_id = str(uuid4())

    suffix = source_path.suffix.lower() or ".jpg"
    stored_filename = f"{image_id}{suffix}"
    stored_path = settings.uploads_dir / stored_filename

    shutil.copy2(source_path, stored_path)

    with Image.open(stored_path) as pil:
        pil = pil.convert("RGB")
        width, height = pil.size

        regions, detector_info = detector.detect(
            image_path=stored_path,
            image_id=image_id,
            width=width,
            height=height,
        )

        context = build_context(
            image=pil,
            prompt=record.prompt,
            regions=regions,
            confidence_threshold=settings.detection_confidence,
        )

    annotated_filename = f"{image_id}.jpg"
    annotated_path = settings.annotated_dir / annotated_filename

    create_annotated_preview(
        stored_path,
        annotated_path,
        regions,
    )

    save_analysis(
        image_id=image_id,
        dataset_record=record.to_dict(),
        stored_filename=stored_filename,
        width=width,
        height=height,
        context=context,
        detector=detector_info,
        annotated_filename=annotated_filename,
        regions=regions,
    )

    saved = get_analysis(image_id)
    assert saved is not None

    return analysis_out(saved)


@app.get(
    f"{settings.api_prefix}/images/{{image_id}}",
    response_model=AnalysisOut,
)
def retrieve_analysis(image_id: str):
    record = get_analysis(image_id)

    if not record:
        raise HTTPException(
            status_code=404,
            detail="Image analysis not found.",
        )

    return analysis_out(record)


@app.get(
    f"{settings.api_prefix}/images/{{image_id}}/handoff",
    response_model=HandoffOut,
)
def module_2_handoff(image_id: str):
    record = get_analysis(image_id)

    if not record:
        raise HTTPException(
            status_code=404,
            detail="Image analysis not found.",
        )

    result = analysis_out(record)

    return HandoffOut(
        image_id=result.image_id,
        dataset_image_name=result.dataset_image_name,
        original_image_url=result.original_image_url,
        prompt=result.prompt,
        generator=result.generator,
        agiqa3k_ground_truth={
            "adj1": result.adj1,
            "adj2": result.adj2,
            "style": result.style,
            "mos_quality": result.mos_quality,
            "std_quality": result.std_quality,
            "mos_align": result.mos_align,
            "std_align": result.std_align,
        },
        regions=result.regions,
    )
