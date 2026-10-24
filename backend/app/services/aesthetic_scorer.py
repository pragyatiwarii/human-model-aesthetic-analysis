from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from app.core.config import settings
from app.services.agiqa3k import load_records


FEATURE_NAMES = [
    "luminance_mean",
    "luminance_std",
    "saturation_mean",
    "saturation_std",
    "colorfulness",
    "gradient_energy",
    "edge_density",
    "entropy",
    "dark_fraction",
    "bright_fraction",
    "center_luminance_delta",
    "horizontal_symmetry",
]

MODEL_VERSION = "agiqa-ridge-v1"


@dataclass(frozen=True)
class LoadedModel:
    version: str
    feature_names: list[str]
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    target_min: float
    target_max: float
    metrics: dict[str, Any]


def _resample() -> int:
    return getattr(Image, "Resampling", Image).BILINEAR


def _prepare_rgb(image: Image.Image, max_side: int = 256) -> np.ndarray:
    rgb = image.convert("RGB")
    width, height = rgb.size
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        rgb = rgb.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            _resample(),
        )
    return np.asarray(rgb, dtype=np.float32) / 255.0


def extract_features(image: Image.Image) -> dict[str, float]:
    arr = _prepare_rgb(image)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b

    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    saturation = np.where(maxc > 1e-6, (maxc - minc) / maxc, 0.0)

    rg = r - g
    yb = 0.5 * (r + g) - b
    colorfulness = math.sqrt(float(rg.std()) ** 2 + float(yb.std()) ** 2)
    colorfulness += 0.3 * math.sqrt(float(rg.mean()) ** 2 + float(yb.mean()) ** 2)

    dx = np.abs(np.diff(lum, axis=1)) if lum.shape[1] > 1 else np.zeros_like(lum)
    dy = np.abs(np.diff(lum, axis=0)) if lum.shape[0] > 1 else np.zeros_like(lum)
    gradient_energy = 0.5 * (float(dx.mean()) + float(dy.mean()))

    if lum.shape[0] > 1 and lum.shape[1] > 1:
        gx = np.pad(dx, ((0, 0), (0, 1)), mode="edge")
        gy = np.pad(dy, ((0, 1), (0, 0)), mode="edge")
        grad = 0.5 * (gx + gy)
        edge_density = float((grad > 0.075).mean())
    else:
        edge_density = 0.0

    hist, _ = np.histogram(lum, bins=64, range=(0.0, 1.0))
    probs = hist.astype(np.float64)
    probs /= max(1.0, probs.sum())
    nz = probs[probs > 0]
    entropy = float(-(nz * np.log2(nz)).sum() / math.log2(64))

    h, w = lum.shape
    y1, y2 = h // 4, max(h // 4 + 1, (3 * h) // 4)
    x1, x2 = w // 4, max(w // 4 + 1, (3 * w) // 4)
    center_mean = float(lum[y1:y2, x1:x2].mean())
    center_delta = abs(center_mean - float(lum.mean()))

    half = w // 2
    if half > 0:
        left = lum[:, :half]
        right = lum[:, -half:][:, ::-1]
        symmetry = 1.0 - float(np.abs(left - right).mean())
    else:
        symmetry = 1.0

    values = {
        "luminance_mean": float(lum.mean()),
        "luminance_std": float(lum.std()),
        "saturation_mean": float(saturation.mean()),
        "saturation_std": float(saturation.std()),
        "colorfulness": float(colorfulness),
        "gradient_energy": float(gradient_energy),
        "edge_density": float(edge_density),
        "entropy": float(entropy),
        "dark_fraction": float((lum < 0.05).mean()),
        "bright_fraction": float((lum > 0.95).mean()),
        "center_luminance_delta": float(center_delta),
        "horizontal_symmetry": float(np.clip(symmetry, 0.0, 1.0)),
    }
    return values


def feature_vector(features: dict[str, float]) -> np.ndarray:
    return np.asarray([features[name] for name in FEATURE_NAMES], dtype=np.float64)


def _hash_int(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()[:8]
    return int.from_bytes(digest, "big", signed=False)


def deterministic_subset(records: Iterable[Any], limit: int) -> list[Any]:
    available = [record for record in records if record.image_exists]
    available.sort(key=lambda record: (_hash_int(record.name), record.name))
    if limit > 0:
        return available[:limit]
    return available


def _fit_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    ridge_lambda: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    xz = (x_train - mean) / scale

    y_mean = float(y_train.mean())
    centered_y = y_train - y_mean
    identity = np.eye(xz.shape[1], dtype=np.float64)
    coefficients = np.linalg.solve(
        xz.T @ xz + ridge_lambda * identity,
        xz.T @ centered_y,
    )
    return coefficients, y_mean, mean, scale


def _predict_array(
    x: np.ndarray,
    coefficients: np.ndarray,
    intercept: float,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return ((x - mean) / scale) @ coefficients + intercept


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - float(np.sum(diff * diff)) / denom if denom > 1e-12 else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2}


def train_model(*, force: bool = False) -> dict[str, Any]:
    path = settings.aesthetic_model_path
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))

    records = deterministic_subset(load_records(), settings.aesthetic_train_limit)
    if len(records) < 40:
        raise RuntimeError(
            "At least 40 AGIQA-3K images are required to train the local aesthetic model."
        )

    xs: list[np.ndarray] = []
    ys: list[float] = []
    names: list[str] = []

    for record in records:
        image_path = settings.agiqa_images_dir / record.name
        try:
            with Image.open(image_path) as image:
                features = extract_features(image)
        except (OSError, ValueError):
            continue
        xs.append(feature_vector(features))
        ys.append(float(record.mos_quality))
        names.append(record.name)

    if len(xs) < 40:
        raise RuntimeError("Too few readable AGIQA-3K images were available for training.")

    x = np.vstack(xs)
    y = np.asarray(ys, dtype=np.float64)

    val_mask = np.asarray([_hash_int(name) % 5 == 0 for name in names], dtype=bool)
    if val_mask.sum() < 8 or (~val_mask).sum() < 20:
        val_mask[:] = False
        val_mask[::5] = True

    x_train, y_train = x[~val_mask], y[~val_mask]
    x_val, y_val = x[val_mask], y[val_mask]

    coefficients, intercept, mean, scale = _fit_ridge(
        x_train,
        y_train,
        settings.aesthetic_ridge_lambda,
    )
    pred_val = _predict_array(x_val, coefficients, intercept, mean, scale)
    metrics = _metrics(y_val, pred_val)

    payload = {
        "version": MODEL_VERSION,
        "model_type": "ridge regression on deterministic visual features",
        "target": "AGIQA-3K mos_quality",
        "score_scale": [float(y.min()), float(y.max())],
        "feature_names": FEATURE_NAMES,
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "coefficients": coefficients.tolist(),
        "intercept": float(intercept),
        "ridge_lambda": float(settings.aesthetic_ridge_lambda),
        "training_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "source_images_used": int(len(y)),
        "training_limit": int(settings.aesthetic_train_limit),
        "validation_metrics": metrics,
        "notes": (
            "Local interpretable aesthetic regressor trained from AGIQA-3K quality MOS. "
            "It is a project-specific model, not a general-purpose aesthetic foundation model."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_model() -> LoadedModel | None:
    path = settings.aesthetic_model_path
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return LoadedModel(
        version=raw["version"],
        feature_names=list(raw["feature_names"]),
        mean=np.asarray(raw["feature_mean"], dtype=np.float64),
        scale=np.asarray(raw["feature_scale"], dtype=np.float64),
        coefficients=np.asarray(raw["coefficients"], dtype=np.float64),
        intercept=float(raw["intercept"]),
        target_min=float(raw["score_scale"][0]),
        target_max=float(raw["score_scale"][1]),
        metrics=raw,
    )


def model_status() -> dict[str, Any]:
    model = load_model()
    if model is None:
        return {
            "trained": False,
            "version": MODEL_VERSION,
            "model_path": str(settings.aesthetic_model_path),
            "training_limit": settings.aesthetic_train_limit,
        }
    return {
        "trained": True,
        "version": model.version,
        "model_path": str(settings.aesthetic_model_path),
        "training_limit": settings.aesthetic_train_limit,
        "training_samples": model.metrics.get("training_samples"),
        "validation_samples": model.metrics.get("validation_samples"),
        "source_images_used": model.metrics.get("source_images_used"),
        "validation_metrics": model.metrics.get("validation_metrics"),
        "target": model.metrics.get("target"),
        "notes": model.metrics.get("notes"),
    }


def score_image(image_path: Path, model: LoadedModel | None = None) -> tuple[float, dict[str, float]]:
    model = model or load_model()
    if model is None:
        raise RuntimeError("Aesthetic model is not trained. Train Module 3 first.")

    with Image.open(image_path) as image:
        features = extract_features(image)
    x = feature_vector(features)
    pred = float(((x - model.mean) / model.scale) @ model.coefficients + model.intercept)
    pred = float(np.clip(pred, model.target_min, model.target_max))
    return pred, features
