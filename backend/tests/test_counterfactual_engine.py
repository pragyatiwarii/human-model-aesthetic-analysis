from pathlib import Path

import numpy as np
from PIL import Image

from app.services.counterfactual_engine import (
    PERTURBATIONS,
    generate_variant,
    public_specs,
)


def gradient_image(size=(40, 30)) -> Image.Image:
    width, height = size
    yy, xx = np.mgrid[0:height, 0:width]
    arr = np.stack(
        [
            (xx * 5 + 30) % 256,
            (yy * 7 + 40) % 256,
            ((xx + yy) * 3 + 60) % 256,
        ],
        axis=2,
    ).astype(np.uint8)
    return Image.fromarray(arr)


def square_region() -> dict:
    return {
        "region_id": "img-r001",
        "label": "person",
        "confidence": 0.9,
        "bbox": [0.25, 0.2, 0.75, 0.8],
        "polygon": [
            [0.25, 0.2],
            [0.75, 0.2],
            [0.75, 0.8],
            [0.25, 0.8],
        ],
        "mask_filename": None,
        "is_fallback": False,
    }


def test_public_specs_have_three_severities():
    specs = public_specs()
    assert len(specs) == len(PERTURBATIONS) == 9
    for spec in specs:
        assert set(spec["severities"]) == {"subtle", "moderate", "strong"}
        assert spec["region_controlled"] is True


def test_region_isolation_is_exact_with_png(tmp_path: Path):
    original_path = tmp_path / "original.png"
    output_path = tmp_path / "counterfactual.png"
    gradient_image().save(original_path)

    params, validation = generate_variant(
        original_path=original_path,
        image_id="img",
        region=square_region(),
        perturbation_type="brightness",
        severity="strong",
        output_path=output_path,
        seed=123,
    )

    assert params["factor"] == 0.48
    assert output_path.suffix == ".png"
    assert validation["lossless_output"] is True
    assert validation["outside_region_unchanged"] is True
    assert validation["outside_region_max_abs_diff"] == 0
    assert validation["inside_region_mean_abs_diff"] > 0
    assert validation["inside_region_changed_fraction"] > 0
    assert validation["mask_source"] == "polygon"


def test_full_image_fallback_modifies_whole_image(tmp_path: Path):
    original_path = tmp_path / "original.png"
    output_path = tmp_path / "full.png"
    gradient_image().save(original_path)

    fallback = {
        "region_id": "img-r001",
        "label": "full_image",
        "confidence": 1.0,
        "bbox": [0.0, 0.0, 1.0, 1.0],
        "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
        "mask_filename": None,
        "is_fallback": True,
    }

    _, validation = generate_variant(
        original_path=original_path,
        image_id="img",
        region=fallback,
        perturbation_type="saturation",
        severity="moderate",
        output_path=output_path,
        seed=456,
    )

    assert validation["mask_source"] == "full_image"
    assert validation["region_pixel_fraction"] == 1.0
    assert validation["outside_region_unchanged"] is True


def test_noise_is_reproducible_for_same_seed(tmp_path: Path):
    original_path = tmp_path / "original.png"
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    gradient_image().save(original_path)

    kwargs = dict(
        original_path=original_path,
        image_id="img",
        region=square_region(),
        perturbation_type="noise",
        severity="moderate",
        seed=987654,
    )

    first_params, _ = generate_variant(output_path=first_path, **kwargs)
    second_params, _ = generate_variant(output_path=second_path, **kwargs)

    first = np.asarray(Image.open(first_path).convert("RGB"))
    second = np.asarray(Image.open(second_path).convert("RGB"))

    assert first_params["seed"] == second_params["seed"] == 987654
    assert np.array_equal(first, second)


def test_all_perturbations_preserve_outside_region(tmp_path: Path):
    original_path = tmp_path / "original.png"
    gradient_image((48, 36)).save(original_path)

    for index, perturbation_type in enumerate(PERTURBATIONS):
        output_path = tmp_path / f"{perturbation_type}.png"
        _, validation = generate_variant(
            original_path=original_path,
            image_id="img",
            region=square_region(),
            perturbation_type=perturbation_type,
            severity="moderate",
            output_path=output_path,
            seed=1000 + index,
        )
        assert validation["outside_region_unchanged"] is True, perturbation_type
        assert validation["outside_region_max_abs_diff"] == 0, perturbation_type
        assert output_path.exists(), perturbation_type
