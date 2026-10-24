from __future__ import annotations

import hashlib
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.core.config import DATA_DIR, settings
from app.db import (
    get_aesthetic_scores,
    get_analysis,
    get_counterfactual_batch,
    get_human_preferences,
    get_region,
    init_db,
    save_aesthetic_scores,
    save_analysis,
    save_counterfactuals,
    save_human_preference,
)
from app.schemas import (
    AestheticBatchScoresOut,
    AestheticModelStatusOut,
    AestheticScoreOut,
    AestheticTrainIn,
    AnalysisOut,
    CounterfactualBatchOut,
    CounterfactualGenerateIn,
    CounterfactualVariantOut,
    DatasetListOut,
    DatasetRecordOut,
    HandoffOut,
    HumanPreferenceIn,
    HumanPreferenceOut,
    Module3HandoffOut,
    Module4HandoffOut,
    PerturbationSpecOut,
    RegionOut,
)
from app.services.agiqa3k import get_record, list_records, status
from app.services.context import build_context
from app.services.aesthetic_scorer import load_model, model_status, score_image, train_model
from app.services.counterfactual_engine import PERTURBATIONS, generate_variant, public_specs
from app.services.image_utils import create_annotated_preview
from app.services.region_detector import RegionDetector


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title=settings.app_name,
    version="3.0.0-module-3",
    description=(
        "Modules 1-2 plus Module 3 aesthetic scoring and human preference "
        "collection for Muskan Dubey."
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


def ground_truth(record: dict) -> dict:
    return {
        "adj1": record.get("adj1"),
        "adj2": record.get("adj2"),
        "style": record.get("style"),
        "mos_quality": record.get("mos_quality"),
        "std_quality": record.get("std_quality"),
        "mos_align": record.get("mos_align"),
        "std_align": record.get("std_align"),
    }


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


def counterfactual_url(filename: str) -> str:
    normalized = filename.replace("\\", "/")
    return f"/media/runtime/counterfactuals/{normalized}"


def batch_out(batch_id: str) -> CounterfactualBatchOut:
    rows = get_counterfactual_batch(batch_id)
    if not rows:
        raise HTTPException(
            status_code=404,
            detail="Counterfactual batch not found.",
        )

    first = rows[0]
    analysis = get_analysis(first["image_id"])
    region = get_region(first["image_id"], first["region_id"])

    if analysis is None or region is None:
        raise HTTPException(
            status_code=409,
            detail="Counterfactual batch references missing Module 1 data.",
        )

    return CounterfactualBatchOut(
        batch_id=batch_id,
        original_image_id=analysis["image_id"],
        original_image_url=(
            f'/media/runtime/uploads/{analysis["stored_filename"]}'
        ),
        dataset_image_name=analysis["dataset_image_name"],
        prompt=analysis["prompt"],
        generator=analysis["generator"],
        agiqa3k_ground_truth=ground_truth(analysis),
        region=region_out(analysis["image_id"], region),
        perturbation_type=first["perturbation_type"],
        variants=[
            CounterfactualVariantOut(
                counterfactual_id=row["counterfactual_id"],
                batch_id=row["batch_id"],
                image_id=row["image_id"],
                region_id=row["region_id"],
                region_label=region["label"],
                perturbation_type=row["perturbation_type"],
                severity=row["severity"],
                parameters=row["parameters"],
                counterfactual_url=counterfactual_url(row["filename"]),
                validation=row["validation"],
            )
            for row in rows
        ],
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "module": 3,
        "module_1_owner": "Vennela Laxmi Puli",
        "module_2_owner": "Sanyojita Priyadarshani Singh",
        "module_3_owner": "Muskan Dubey",
        "dataset": "AGIQA-3K",
        "caption_model": "not used",
        "detection_confidence": settings.detection_confidence,
        "ml_models_enabled": settings.enable_ml_models,
        "counterfactual_output": "lossless PNG",
        "perturbation_count": len(PERTURBATIONS),
        "aesthetic_model": model_status(),
        "human_preference_collection": True,
    }


# -------------------------- Module 1 ---------------------------------


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


# -------------------------- Module 2 ---------------------------------


@app.get(
    f"{settings.api_prefix}/perturbations",
    response_model=list[PerturbationSpecOut],
)
def list_perturbations():
    return public_specs()


@app.post(
    f"{settings.api_prefix}/counterfactuals/generate",
    response_model=CounterfactualBatchOut,
)
def generate_counterfactuals(payload: CounterfactualGenerateIn):
    analysis = get_analysis(payload.image_id)
    if not analysis:
        raise HTTPException(
            status_code=404,
            detail="Module 1 image analysis not found. Run Module 1 first.",
        )

    region = get_region(payload.image_id, payload.region_id)
    if not region:
        raise HTTPException(
            status_code=404,
            detail="Selected region does not belong to this image.",
        )

    if payload.perturbation_type not in PERTURBATIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported perturbation: {payload.perturbation_type}",
        )

    requested: list[str] = []
    for severity in payload.severities:
        if severity not in requested:
            requested.append(severity)

    original_path = settings.uploads_dir / analysis["stored_filename"]
    if not original_path.exists():
        raise HTTPException(
            status_code=409,
            detail="Stored Module 1 original image is missing.",
        )

    batch_id = str(uuid4())
    output_dir = settings.counterfactuals_dir / payload.image_id / batch_id

    rows: list[dict] = []

    try:
        for severity in requested:
            counterfactual_id = str(uuid4())
            filename = f"{severity}.png"
            relative_filename = str(
                Path(payload.image_id) / batch_id / filename
            )
            output_path = output_dir / filename

            seed_bytes = hashlib.sha256(
                f"{batch_id}:{severity}".encode("utf-8")
            ).digest()[:4]
            seed = int.from_bytes(seed_bytes, "big", signed=False)

            parameters, validation = generate_variant(
                original_path=original_path,
                image_id=payload.image_id,
                region=region,
                perturbation_type=payload.perturbation_type,
                severity=severity,
                output_path=output_path,
                seed=seed,
            )

            rows.append(
                {
                    "counterfactual_id": counterfactual_id,
                    "batch_id": batch_id,
                    "image_id": payload.image_id,
                    "region_id": payload.region_id,
                    "perturbation_type": payload.perturbation_type,
                    "severity": severity,
                    "filename": relative_filename,
                    "parameters": parameters,
                    "validation": validation,
                }
            )
    except (ValueError, RuntimeError) as exc:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    save_counterfactuals(rows)
    return batch_out(batch_id)


@app.get(
    f"{settings.api_prefix}/counterfactuals/batches/{{batch_id}}",
    response_model=CounterfactualBatchOut,
)
def retrieve_counterfactual_batch(batch_id: str):
    return batch_out(batch_id)


@app.get(
    f"{settings.api_prefix}/counterfactuals/batches/{{batch_id}}/handoff",
    response_model=Module3HandoffOut,
)
def module_3_handoff(batch_id: str):
    batch = batch_out(batch_id)

    spec = PERTURBATIONS[batch.perturbation_type]

    return Module3HandoffOut(
        batch_id=batch.batch_id,
        original={
            "image_id": batch.original_image_id,
            "dataset_name": "AGIQA-3K",
            "dataset_image_name": batch.dataset_image_name,
            "image_url": batch.original_image_url,
            "prompt": batch.prompt,
            "generator": batch.generator,
            "agiqa3k_ground_truth": batch.agiqa3k_ground_truth,
        },
        region=batch.region.model_dump(),
        perturbation={
            "type": batch.perturbation_type,
            "label": spec["label"],
            "description": spec["description"],
            "region_controlled": True,
        },
        counterfactuals=[
            {
                "counterfactual_id": variant.counterfactual_id,
                "severity": variant.severity,
                "parameters": variant.parameters,
                "image_url": variant.counterfactual_url,
                "validation": variant.validation,
            }
            for variant in batch.variants
        ],
    )

# -------------------------- Module 3 ---------------------------------


def _batch_candidates(batch_id: str) -> tuple[dict, dict, list[dict]]:
    rows = get_counterfactual_batch(batch_id)
    if not rows:
        raise HTTPException(status_code=404, detail="Counterfactual batch not found.")

    analysis = get_analysis(rows[0]["image_id"])
    region = get_region(rows[0]["image_id"], rows[0]["region_id"])
    if analysis is None or region is None:
        raise HTTPException(status_code=409, detail="Module 2 batch references missing upstream data.")

    candidates = [
        {
            "candidate_id": f'original:{analysis["image_id"]}',
            "source_type": "original",
            "severity": None,
            "image_url": f'/media/runtime/uploads/{analysis["stored_filename"]}',
            "image_path": settings.uploads_dir / analysis["stored_filename"],
        }
    ]
    for row in rows:
        candidates.append(
            {
                "candidate_id": row["counterfactual_id"],
                "source_type": "counterfactual",
                "severity": row["severity"],
                "image_url": counterfactual_url(row["filename"]),
                "image_path": settings.counterfactuals_dir / row["filename"],
            }
        )
    return analysis, region, candidates


@app.get(
    f"{settings.api_prefix}/aesthetic-model/status",
    response_model=AestheticModelStatusOut,
)
def aesthetic_model_status():
    return model_status()


@app.post(
    f"{settings.api_prefix}/aesthetic-model/train",
    response_model=AestheticModelStatusOut,
)
def train_aesthetic_model(payload: AestheticTrainIn):
    try:
        train_model(force=payload.force)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return model_status()


@app.post(
    f"{settings.api_prefix}/aesthetic/batches/{{batch_id}}/score",
    response_model=AestheticBatchScoresOut,
)
def score_counterfactual_batch(batch_id: str):
    model = load_model()
    if model is None:
        raise HTTPException(
            status_code=409,
            detail="Aesthetic model is not trained. Train Module 3 first.",
        )

    _analysis, _region, candidates = _batch_candidates(batch_id)
    rows: list[dict] = []
    for candidate in candidates:
        image_path = candidate["image_path"]
        if not image_path.exists():
            raise HTTPException(status_code=409, detail=f'Missing candidate image: {image_path.name}')
        score, features = score_image(image_path, model)
        rows.append(
            {
                "score_id": str(uuid4()),
                "batch_id": batch_id,
                "candidate_id": candidate["candidate_id"],
                "source_type": candidate["source_type"],
                "severity": candidate["severity"],
                "image_url": candidate["image_url"],
                "score": round(score, 6),
                "model_version": model.version,
                "features": features,
            }
        )

    save_aesthetic_scores(rows)
    preferred = max(rows, key=lambda row: row["score"])["candidate_id"]
    return AestheticBatchScoresOut(
        batch_id=batch_id,
        model=model_status(),
        candidates=[AestheticScoreOut(**row) for row in rows],
        model_preferred_candidate_id=preferred,
    )


@app.get(
    f"{settings.api_prefix}/aesthetic/batches/{{batch_id}}/scores",
    response_model=AestheticBatchScoresOut,
)
def retrieve_batch_scores(batch_id: str):
    rows = get_aesthetic_scores(batch_id)
    if not rows:
        raise HTTPException(status_code=404, detail="No aesthetic scores found for this batch.")
    preferred = max(rows, key=lambda row: row["score"])["candidate_id"]
    return AestheticBatchScoresOut(
        batch_id=batch_id,
        model=model_status(),
        candidates=[AestheticScoreOut(**row) for row in rows],
        model_preferred_candidate_id=preferred,
    )


@app.post(
    f"{settings.api_prefix}/human-preferences",
    response_model=HumanPreferenceOut,
)
def submit_human_preference(payload: HumanPreferenceIn):
    _analysis, _region, candidates = _batch_candidates(payload.batch_id)
    candidate_ids = {candidate["candidate_id"] for candidate in candidates}

    if set(payload.ratings) != candidate_ids:
        raise HTTPException(
            status_code=400,
            detail="Provide one 1-5 rating for every original/counterfactual candidate.",
        )
    invalid = [key for key, value in payload.ratings.items() if not 1 <= int(value) <= 5]
    if invalid:
        raise HTTPException(status_code=400, detail="All aesthetic ratings must be integers from 1 to 5.")
    if payload.preferred_candidate_id not in candidate_ids:
        raise HTTPException(status_code=400, detail="Preferred candidate is not part of this batch.")

    row = {
        "preference_id": str(uuid4()),
        "batch_id": payload.batch_id,
        "participant_code": payload.participant_code.strip(),
        "ratings": {key: int(value) for key, value in payload.ratings.items()},
        "preferred_candidate_id": payload.preferred_candidate_id,
        "notes": payload.notes.strip() if payload.notes else None,
    }
    save_human_preference(row)
    stored = get_human_preferences(payload.batch_id)[-1]
    return HumanPreferenceOut(**stored)


@app.get(
    f"{settings.api_prefix}/human-preferences/batches/{{batch_id}}",
    response_model=list[HumanPreferenceOut],
)
def retrieve_human_preferences(batch_id: str):
    _batch_candidates(batch_id)
    return [HumanPreferenceOut(**row) for row in get_human_preferences(batch_id)]


@app.get(
    f"{settings.api_prefix}/module3/batches/{{batch_id}}/handoff",
    response_model=Module4HandoffOut,
)
def module_4_handoff(batch_id: str):
    analysis, region, candidates = _batch_candidates(batch_id)
    score_rows = get_aesthetic_scores(batch_id)
    preferences = get_human_preferences(batch_id)

    if not score_rows:
        raise HTTPException(status_code=409, detail="Score the batch with Module 3 before Module 4 handoff.")
    if not preferences:
        raise HTTPException(status_code=409, detail="Collect at least one human preference before Module 4 handoff.")

    score_by_id = {row["candidate_id"]: row for row in score_rows}
    candidate_payload = []
    for candidate in candidates:
        score = score_by_id.get(candidate["candidate_id"])
        candidate_payload.append(
            {
                "candidate_id": candidate["candidate_id"],
                "source_type": candidate["source_type"],
                "severity": candidate["severity"],
                "image_url": candidate["image_url"],
                "model_aesthetic_score": score["score"] if score else None,
                "model_version": score["model_version"] if score else None,
            }
        )

    average_rating: dict[str, float] = {}
    preferred_count: dict[str, int] = {candidate["candidate_id"]: 0 for candidate in candidates}
    for candidate in candidates:
        values = [p["ratings"][candidate["candidate_id"]] for p in preferences]
        average_rating[candidate["candidate_id"]] = round(sum(values) / len(values), 4)
    for preference in preferences:
        preferred_count[preference["preferred_candidate_id"]] += 1

    first_batch_row = get_counterfactual_batch(batch_id)[0]
    spec = PERTURBATIONS[first_batch_row["perturbation_type"]]
    model_preferred = max(score_rows, key=lambda row: row["score"])["candidate_id"]

    return Module4HandoffOut(
        batch_id=batch_id,
        original={
            "image_id": analysis["image_id"],
            "dataset_name": "AGIQA-3K",
            "dataset_image_name": analysis["dataset_image_name"],
            "prompt": analysis["prompt"],
            "generator": analysis["generator"],
            "agiqa3k_ground_truth": ground_truth(analysis),
        },
        region=region_out(analysis["image_id"], region).model_dump(),
        perturbation={
            "type": first_batch_row["perturbation_type"],
            "label": spec["label"],
            "description": spec["description"],
        },
        model={
            **model_status(),
            "model_preferred_candidate_id": model_preferred,
        },
        candidates=candidate_payload,
        human_preferences=preferences,
        human_summary={
            "participant_count": len(preferences),
            "average_rating_by_candidate": average_rating,
            "preferred_count_by_candidate": preferred_count,
        },
    )

