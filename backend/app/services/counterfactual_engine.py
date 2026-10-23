from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from app.core.config import settings


PERTURBATIONS: dict[str, dict[str, Any]] = {
    "brightness": {
        "label": "Brightness reduction",
        "description": "Darkens only the selected region.",
        "severities": {
            "subtle": {"factor": 0.88},
            "moderate": {"factor": 0.70},
            "strong": {"factor": 0.48},
        },
    },
    "contrast": {
        "label": "Contrast reduction",
        "description": "Reduces contrast only inside the selected region.",
        "severities": {
            "subtle": {"factor": 0.86},
            "moderate": {"factor": 0.62},
            "strong": {"factor": 0.38},
        },
    },
    "saturation": {
        "label": "Saturation reduction",
        "description": "Desaturates only the selected region.",
        "severities": {
            "subtle": {"factor": 0.78},
            "moderate": {"factor": 0.46},
            "strong": {"factor": 0.14},
        },
    },
    "blur": {
        "label": "Gaussian blur",
        "description": "Applies Gaussian blur only inside the selected region.",
        "severities": {
            "subtle": {"radius": 1.2},
            "moderate": {"radius": 3.0},
            "strong": {"radius": 6.0},
        },
    },
    "noise": {
        "label": "Gaussian noise",
        "description": "Adds deterministic Gaussian noise only inside the selected region.",
        "severities": {
            "subtle": {"sigma": 8.0},
            "moderate": {"sigma": 20.0},
            "strong": {"sigma": 38.0},
        },
    },
    "sharpness": {
        "label": "Over-sharpening",
        "description": "Increases sharpness only inside the selected region.",
        "severities": {
            "subtle": {"factor": 1.8},
            "moderate": {"factor": 3.0},
            "strong": {"factor": 5.0},
        },
    },
    "color_shift": {
        "label": "Colour / hue shift",
        "description": "Rotates hue only inside the selected region.",
        "severities": {
            "subtle": {"hue_degrees": 12},
            "moderate": {"hue_degrees": 32},
            "strong": {"hue_degrees": 65},
        },
    },
    "pixelation": {
        "label": "Pixelation",
        "description": "Reduces spatial detail only inside the selected region.",
        "severities": {
            "subtle": {"scale": 0.62},
            "moderate": {"scale": 0.32},
            "strong": {"scale": 0.16},
        },
    },
    "structural_distortion": {
        "label": "Structural distortion",
        "description": "Shifts alternating horizontal bands inside the selected region.",
        "severities": {
            "subtle": {"shift_ratio": 0.025, "bands": 8},
            "moderate": {"shift_ratio": 0.055, "bands": 8},
            "strong": {"shift_ratio": 0.10, "bands": 10},
        },
    },
}


def public_specs() -> list[dict[str, Any]]:
    return [
        {
            "key": key,
            "label": value["label"],
            "description": value["description"],
            "region_controlled": True,
            "severities": value["severities"],
        }
        for key, value in PERTURBATIONS.items()
    ]


def _bbox_pixels(
    bbox: list[float],
    size: tuple[int, int],
) -> tuple[int, int, int, int]:
    width, height = size
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(width - 1, int(round(x1 * width)))),
        max(0, min(height - 1, int(round(y1 * height)))),
        max(1, min(width, int(round(x2 * width)))),
        max(1, min(height, int(round(y2 * height)))),
    )


def build_region_mask(
    image_id: str,
    region: dict[str, Any],
    size: tuple[int, int],
) -> tuple[Image.Image, str]:
    """Return a binary region mask and the source used to construct it."""
    width, height = size

    if region.get("is_fallback"):
        return Image.new("L", size, 255), "full_image"

    mask_filename = region.get("mask_filename")
    if mask_filename:
        mask_path = settings.regions_dir / image_id / mask_filename
        if mask_path.exists():
            with Image.open(mask_path) as stored:
                mask = stored.convert("L").resize(size, Image.Resampling.NEAREST)
            binary = mask.point(lambda value: 255 if value >= 128 else 0)
            return binary, "segmentation_mask"

    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)

    polygon = region.get("polygon")
    if polygon and len(polygon) >= 3:
        points = [
            (
                int(round(max(0.0, min(1.0, x)) * (width - 1))),
                int(round(max(0.0, min(1.0, y)) * (height - 1))),
            )
            for x, y in polygon
        ]
        draw.polygon(points, fill=255)
        return mask, "polygon"

    draw.rectangle(_bbox_pixels(region["bbox"], size), fill=255)
    return mask, "bbox"


def _hue_shift(image: Image.Image, degrees: float) -> Image.Image:
    hsv = np.array(image.convert("HSV"), dtype=np.uint8)
    hue_delta = int(round((degrees / 360.0) * 255.0))
    hue = hsv[..., 0].astype(np.int16)
    hsv[..., 0] = ((hue + hue_delta) % 256).astype(np.uint8)
    return Image.fromarray(hsv, "HSV").convert("RGB")


def _pixelate(image: Image.Image, scale: float) -> Image.Image:
    width, height = image.size
    small_width = max(1, int(round(width * scale)))
    small_height = max(1, int(round(height * scale)))
    return image.resize(
        (small_width, small_height),
        Image.Resampling.BILINEAR,
    ).resize((width, height), Image.Resampling.NEAREST)


def _structural_distortion(
    image: Image.Image,
    bbox: list[float],
    shift_ratio: float,
    bands: int,
) -> Image.Image:
    arr = np.asarray(image.convert("RGB")).copy()
    x1, y1, x2, y2 = _bbox_pixels(bbox, image.size)
    crop = arr[y1:y2, x1:x2].copy()

    if crop.size == 0:
        return image.copy()

    height, width = crop.shape[:2]
    band_height = max(1, height // max(1, bands))
    shift = max(1, int(round(width * shift_ratio)))

    for index, start in enumerate(range(0, height, band_height)):
        end = min(height, start + band_height)
        direction = 1 if index % 2 == 0 else -1
        crop[start:end] = np.roll(
            crop[start:end],
            direction * shift,
            axis=1,
        )

    arr[y1:y2, x1:x2] = crop
    return Image.fromarray(arr, "RGB")


def make_candidate(
    original: Image.Image,
    perturbation_type: str,
    params: dict[str, Any],
    bbox: list[float],
    seed: int,
) -> Image.Image:
    original = original.convert("RGB")

    if perturbation_type == "brightness":
        return ImageEnhance.Brightness(original).enhance(params["factor"])
    if perturbation_type == "contrast":
        return ImageEnhance.Contrast(original).enhance(params["factor"])
    if perturbation_type == "saturation":
        return ImageEnhance.Color(original).enhance(params["factor"])
    if perturbation_type == "blur":
        return original.filter(ImageFilter.GaussianBlur(radius=params["radius"]))
    if perturbation_type == "sharpness":
        return ImageEnhance.Sharpness(original).enhance(params["factor"])
    if perturbation_type == "color_shift":
        return _hue_shift(original, params["hue_degrees"])
    if perturbation_type == "pixelation":
        return _pixelate(original, params["scale"])
    if perturbation_type == "structural_distortion":
        return _structural_distortion(
            original,
            bbox,
            params["shift_ratio"],
            params["bands"],
        )
    if perturbation_type == "noise":
        rng = np.random.default_rng(seed)
        arr = np.asarray(original, dtype=np.float32)
        noise = rng.normal(0.0, params["sigma"], size=arr.shape)
        return Image.fromarray(
            np.clip(arr + noise, 0, 255).astype(np.uint8),
            "RGB",
        )

    raise ValueError(f"Unsupported perturbation: {perturbation_type}")


def _validate_isolation(
    original: Image.Image,
    result: Image.Image,
    mask: Image.Image,
) -> dict[str, Any]:
    before = np.asarray(original.convert("RGB"), dtype=np.int16)
    after = np.asarray(result.convert("RGB"), dtype=np.int16)
    mask_arr = np.asarray(mask.convert("L")) > 0

    abs_diff = np.abs(after - before)
    changed = np.any(abs_diff > 0, axis=2)

    inside_count = int(mask_arr.sum())
    outside = ~mask_arr
    outside_count = int(outside.sum())

    outside_max = int(abs_diff[outside].max()) if outside_count else 0
    inside_mean = (
        float(abs_diff[mask_arr].mean()) if inside_count else 0.0
    )
    changed_inside = (
        float(changed[mask_arr].mean()) if inside_count else 0.0
    )
    region_fraction = float(mask_arr.mean())

    return {
        "lossless_output": True,
        "outside_region_max_abs_diff": outside_max,
        "outside_region_unchanged": outside_max == 0,
        "inside_region_mean_abs_diff": round(inside_mean, 4),
        "inside_region_changed_fraction": round(changed_inside, 6),
        "region_pixel_fraction": round(region_fraction, 6),
    }


def generate_variant(
    *,
    original_path: Path,
    image_id: str,
    region: dict[str, Any],
    perturbation_type: str,
    severity: str,
    output_path: Path,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if perturbation_type not in PERTURBATIONS:
        raise ValueError(f"Unsupported perturbation: {perturbation_type}")

    severity_map = PERTURBATIONS[perturbation_type]["severities"]
    if severity not in severity_map:
        raise ValueError(f"Unsupported severity: {severity}")

    params = dict(severity_map[severity])
    if perturbation_type == "noise":
        params["seed"] = int(seed)

    with Image.open(original_path) as source:
        original = source.convert("RGB")

    mask, mask_source = build_region_mask(
        image_id=image_id,
        region=region,
        size=original.size,
    )

    candidate = make_candidate(
        original=original,
        perturbation_type=perturbation_type,
        params=params,
        bbox=region["bbox"],
        seed=seed,
    )

    result = Image.composite(candidate, original, mask)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # PNG is deliberate: JPEG recompression can change pixels outside the mask.
    result.save(output_path, format="PNG", optimize=False)

    # Reopen the saved PNG so validation checks the actual persisted artifact.
    with Image.open(output_path) as persisted:
        persisted_rgb = persisted.convert("RGB")
        validation = _validate_isolation(original, persisted_rgb, mask)

    validation["mask_source"] = mask_source

    if not validation["outside_region_unchanged"]:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Counterfactual isolation check failed: pixels outside the "
            "selected region changed."
        )

    return params, validation
