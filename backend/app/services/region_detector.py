from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from app.core.config import settings


class RegionDetector:
    def __init__(self) -> None:
        self._model = None
        self._load_error: str | None = None

    def _get_model(self):
        if not settings.enable_ml_models:
            return None

        if self._model is not None:
            return self._model

        try:
            from ultralytics import YOLO

            self._model = YOLO(settings.yolo_model)
            return self._model
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            return None

    @staticmethod
    def fallback(image_id: str) -> dict[str, Any]:
        return {
            "region_id": f"{image_id}-r001",
            "label": "full_image",
            "confidence": 1.0,
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "polygon": [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            "mask_filename": None,
            "is_fallback": True,
        }

    @staticmethod
    def _save_mask(
        polygon_px: list[list[float]],
        width: int,
        height: int,
        output_path: Path,
    ) -> None:
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(
            [(int(x), int(y)) for x, y in polygon_px],
            fill=255,
        )
        mask.save(output_path)

    def detect(
        self,
        image_path: Path,
        image_id: str,
        width: int,
        height: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        model = self._get_model()

        if model is None:
            return [self.fallback(image_id)], {
                "enabled": settings.enable_ml_models,
                "model": settings.yolo_model if settings.enable_ml_models else None,
                "confidence_threshold": settings.detection_confidence,
                "error": self._load_error,
                "used_fallback": True,
            }

        try:
            results = model.predict(
                source=str(image_path),
                conf=settings.detection_confidence,
                verbose=False,
            )

            if not results:
                return [self.fallback(image_id)], {
                    "enabled": True,
                    "model": settings.yolo_model,
                    "confidence_threshold": settings.detection_confidence,
                    "error": None,
                    "used_fallback": True,
                }

            result = results[0]

            boxes = (
                result.boxes.xyxy.cpu().tolist()
                if result.boxes is not None
                else []
            )
            confs = (
                result.boxes.conf.cpu().tolist()
                if result.boxes is not None
                else []
            )
            classes = (
                result.boxes.cls.cpu().tolist()
                if result.boxes is not None
                else []
            )
            names = result.names

            polygons_px: list[list[list[float]]] = []
            if result.masks is not None and result.masks.xy is not None:
                polygons_px = [
                    [[float(x), float(y)] for x, y in polygon.tolist()]
                    for polygon in result.masks.xy
                ]

            mask_dir = settings.regions_dir / image_id
            mask_dir.mkdir(parents=True, exist_ok=True)

            regions: list[dict[str, Any]] = []

            for index, box in enumerate(boxes):
                confidence = (
                    float(confs[index])
                    if index < len(confs)
                    else 0.0
                )

                # Defense in depth: never expose weak detections.
                if confidence < settings.detection_confidence:
                    continue

                x1, y1, x2, y2 = box

                class_index = (
                    int(classes[index])
                    if index < len(classes)
                    else -1
                )
                label = str(
                    names.get(class_index, f"class_{class_index}")
                )

                region_id = f"{image_id}-r{len(regions)+1:03d}"

                bbox = [
                    max(0.0, min(1.0, x1 / width)),
                    max(0.0, min(1.0, y1 / height)),
                    max(0.0, min(1.0, x2 / width)),
                    max(0.0, min(1.0, y2 / height)),
                ]

                polygon_px = (
                    polygons_px[index]
                    if index < len(polygons_px)
                    else None
                )

                polygon = None
                mask_filename = None

                if polygon_px and len(polygon_px) >= 3:
                    polygon = [
                        [
                            max(0.0, min(1.0, x / width)),
                            max(0.0, min(1.0, y / height)),
                        ]
                        for x, y in polygon_px
                    ]

                    mask_filename = f"{region_id}.png"
                    self._save_mask(
                        polygon_px,
                        width,
                        height,
                        mask_dir / mask_filename,
                    )

                regions.append(
                    {
                        "region_id": region_id,
                        "label": label,
                        "confidence": round(confidence, 6),
                        "bbox": [round(v, 6) for v in bbox],
                        "polygon": polygon,
                        "mask_filename": mask_filename,
                        "is_fallback": False,
                    }
                )

            if not regions:
                return [self.fallback(image_id)], {
                    "enabled": True,
                    "model": settings.yolo_model,
                    "confidence_threshold": settings.detection_confidence,
                    "error": None,
                    "used_fallback": True,
                }

            return regions, {
                "enabled": True,
                "model": settings.yolo_model,
                "confidence_threshold": settings.detection_confidence,
                "error": None,
                "used_fallback": False,
            }

        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"

            return [self.fallback(image_id)], {
                "enabled": True,
                "model": settings.yolo_model,
                "confidence_threshold": settings.detection_confidence,
                "error": self._load_error,
                "used_fallback": True,
            }
