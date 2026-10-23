from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings


SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    image_id TEXT PRIMARY KEY,
    dataset_image_name TEXT NOT NULL,
    stored_filename TEXT NOT NULL,
    prompt TEXT NOT NULL,
    generator TEXT,
    adj1 TEXT,
    adj2 TEXT,
    style TEXT,
    mos_quality REAL,
    std_quality REAL,
    mos_align REAL,
    std_align REAL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    context_json TEXT NOT NULL,
    detector_json TEXT NOT NULL,
    annotated_filename TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS regions (
    region_id TEXT PRIMARY KEY,
    image_id TEXT NOT NULL,
    label TEXT NOT NULL,
    confidence REAL NOT NULL,
    bbox_json TEXT NOT NULL,
    polygon_json TEXT,
    mask_filename TEXT,
    is_fallback INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (image_id) REFERENCES analyses(image_id)
);
"""


@contextmanager
def connection():
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row

    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connection() as conn:
        conn.executescript(SCHEMA)


def save_analysis(
    *,
    image_id: str,
    dataset_record: dict[str, Any],
    stored_filename: str,
    width: int,
    height: int,
    context: dict[str, Any],
    detector: dict[str, Any],
    annotated_filename: str | None,
    regions: list[dict[str, Any]],
) -> None:
    created_at = datetime.now(timezone.utc).isoformat()

    with connection() as conn:
        conn.execute(
            """
            INSERT INTO analyses (
                image_id,
                dataset_image_name,
                stored_filename,
                prompt,
                generator,
                adj1,
                adj2,
                style,
                mos_quality,
                std_quality,
                mos_align,
                std_align,
                width,
                height,
                context_json,
                detector_json,
                annotated_filename,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                image_id,
                dataset_record["name"],
                stored_filename,
                dataset_record["prompt"],
                dataset_record["generator"],
                dataset_record.get("adj1"),
                dataset_record.get("adj2"),
                dataset_record.get("style"),
                dataset_record["mos_quality"],
                dataset_record["std_quality"],
                dataset_record["mos_align"],
                dataset_record["std_align"],
                width,
                height,
                json.dumps(context),
                json.dumps(detector),
                annotated_filename,
                created_at,
            ),
        )

        conn.executemany(
            """
            INSERT INTO regions (
                region_id,
                image_id,
                label,
                confidence,
                bbox_json,
                polygon_json,
                mask_filename,
                is_fallback
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    region["region_id"],
                    image_id,
                    region["label"],
                    region["confidence"],
                    json.dumps(region["bbox"]),
                    json.dumps(region.get("polygon")),
                    region.get("mask_filename"),
                    1 if region.get("is_fallback") else 0,
                )
                for region in regions
            ],
        )


def get_analysis(image_id: str) -> dict[str, Any] | None:
    with connection() as conn:
        analysis = conn.execute(
            "SELECT * FROM analyses WHERE image_id = ?",
            (image_id,),
        ).fetchone()

        if not analysis:
            return None

        rows = conn.execute(
            "SELECT * FROM regions WHERE image_id = ? ORDER BY region_id",
            (image_id,),
        ).fetchall()

    regions = [
        {
            "region_id": row["region_id"],
            "label": row["label"],
            "confidence": row["confidence"],
            "bbox": json.loads(row["bbox_json"]),
            "polygon": (
                json.loads(row["polygon_json"])
                if row["polygon_json"]
                else None
            ),
            "mask_filename": row["mask_filename"],
            "is_fallback": bool(row["is_fallback"]),
        }
        for row in rows
    ]

    return {
        "image_id": analysis["image_id"],
        "dataset_image_name": analysis["dataset_image_name"],
        "stored_filename": analysis["stored_filename"],
        "prompt": analysis["prompt"],
        "generator": analysis["generator"],
        "adj1": analysis["adj1"],
        "adj2": analysis["adj2"],
        "style": analysis["style"],
        "mos_quality": analysis["mos_quality"],
        "std_quality": analysis["std_quality"],
        "mos_align": analysis["mos_align"],
        "std_align": analysis["std_align"],
        "width": analysis["width"],
        "height": analysis["height"],
        "context": json.loads(analysis["context_json"]),
        "detector": json.loads(analysis["detector_json"]),
        "annotated_filename": analysis["annotated_filename"],
        "created_at": analysis["created_at"],
        "regions": regions,
    }
