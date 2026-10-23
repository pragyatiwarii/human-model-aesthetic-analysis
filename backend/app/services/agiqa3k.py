from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from functools import lru_cache

from app.core.config import settings


REQUIRED_COLUMNS = {
    "name",
    "prompt",
    "adj1",
    "adj2",
    "style",
    "mos_quality",
    "std_quality",
    "mos_align",
    "std_align",
}


@dataclass(frozen=True)
class AGIQARecord:
    name: str
    prompt: str
    adj1: str | None
    adj2: str | None
    style: str | None
    mos_quality: float
    std_quality: float
    mos_align: float
    std_align: float
    generator: str
    image_exists: bool

    def to_dict(self):
        return asdict(self)


def _optional(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None


def infer_generator(filename: str) -> str:
    return filename.split("_", 1)[0]


@lru_cache(maxsize=1)
def load_records() -> list[AGIQARecord]:
    path = settings.agiqa_metadata_path
    if not path.exists():
        return []

    records: list[AGIQARecord] = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(
                "AGIQA-3K data.csv is missing columns: " + ", ".join(missing)
            )

        for row in reader:
            name = row["name"].strip()
            records.append(
                AGIQARecord(
                    name=name,
                    prompt=row["prompt"].strip(),
                    adj1=_optional(row["adj1"]),
                    adj2=_optional(row["adj2"]),
                    style=_optional(row["style"]),
                    mos_quality=float(row["mos_quality"]),
                    std_quality=float(row["std_quality"]),
                    mos_align=float(row["mos_align"]),
                    std_align=float(row["std_align"]),
                    generator=infer_generator(name),
                    image_exists=(settings.agiqa_images_dir / name).exists(),
                )
            )

    return records


def clear_cache() -> None:
    load_records.cache_clear()


def get_record(name: str) -> AGIQARecord | None:
    for record in load_records():
        if record.name == name:
            return record
    return None


def list_records(
    *,
    offset: int = 0,
    limit: int = 25,
    search: str | None = None,
    generator: str | None = None,
) -> tuple[list[AGIQARecord], int]:
    records = load_records()

    if search:
        q = search.strip().lower()
        records = [
            r
            for r in records
            if q in r.name.lower() or q in r.prompt.lower()
        ]

    if generator:
        g = generator.strip().lower()
        records = [r for r in records if r.generator.lower() == g]

    total = len(records)
    return records[offset : offset + limit], total


def status() -> dict:
    records = load_records()
    available = sum(1 for r in records if r.image_exists)

    return {
        "dataset": "AGIQA-3K",
        "metadata_found": settings.agiqa_metadata_path.exists(),
        "metadata_path": str(settings.agiqa_metadata_path),
        "record_count": len(records),
        "available_image_count": available,
        "missing_image_count": max(0, len(records) - available),
        "images_directory": str(settings.agiqa_images_dir),
        "generators": sorted({r.generator for r in records}),
        "ready": bool(records) and available > 0,
    }
