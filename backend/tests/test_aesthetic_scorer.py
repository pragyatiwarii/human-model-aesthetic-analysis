from dataclasses import dataclass

import numpy as np
from PIL import Image

from app.services.aesthetic_scorer import (
    FEATURE_NAMES,
    LoadedModel,
    _fit_ridge,
    _predict_array,
    deterministic_subset,
    extract_features,
    score_image,
)


def test_extract_features_has_finite_expected_values():
    image = Image.new("RGB", (64, 64), (120, 80, 40))
    features = extract_features(image)
    assert list(features) == FEATURE_NAMES
    assert all(np.isfinite(value) for value in features.values())
    assert 0 <= features["luminance_mean"] <= 1
    assert 0 <= features["horizontal_symmetry"] <= 1


def test_ridge_fit_predicts_simple_relation():
    x = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    y = 1.5 + 0.8 * x[:, 0] - 0.3 * x[:, 1]
    coef, intercept, mean, scale = _fit_ridge(x, y, 1e-8)
    pred = _predict_array(x, coef, intercept, mean, scale)
    assert np.max(np.abs(pred - y)) < 1e-5


@dataclass
class FakeRecord:
    name: str
    image_exists: bool = True


def test_deterministic_subset_is_stable():
    records = [FakeRecord(f"img_{index}.jpg") for index in range(20)]
    first = [r.name for r in deterministic_subset(records, 7)]
    second = [r.name for r in deterministic_subset(reversed(records), 7)]
    assert first == second
    assert len(first) == 7


def test_score_image_clips_to_model_scale(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (64, 64), (255, 255, 255)).save(path)
    model = LoadedModel(
        version="test",
        feature_names=FEATURE_NAMES,
        mean=np.zeros(len(FEATURE_NAMES)),
        scale=np.ones(len(FEATURE_NAMES)),
        coefficients=np.ones(len(FEATURE_NAMES)) * 100.0,
        intercept=100.0,
        target_min=1.0,
        target_max=5.0,
        metrics={},
    )
    score, features = score_image(path, model)
    assert score == 5.0
    assert set(features) == set(FEATURE_NAMES)
