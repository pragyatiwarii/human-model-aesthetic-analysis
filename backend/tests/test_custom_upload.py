from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.core.config import settings


def test_custom_image_prompt_can_enter_module1(monkeypatch, tmp_path):
    uploads = tmp_path / "uploads"
    annotated = tmp_path / "annotated"
    regions = tmp_path / "regions"
    for folder in (uploads, annotated, regions):
        folder.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(settings, "db_path", tmp_path / "test.db")
    monkeypatch.setattr(settings, "uploads_dir", uploads)
    monkeypatch.setattr(settings, "annotated_dir", annotated)
    monkeypatch.setattr(settings, "regions_dir", regions)
    monkeypatch.setattr(
        main.detector,
        "detect",
        lambda **kwargs: (
            [main.detector.fallback(kwargs["image_id"])],
            {
                "enabled": False,
                "model": None,
                "confidence_threshold": 0.55,
                "error": None,
                "used_fallback": True,
            },
        ),
    )
    monkeypatch.setattr(
        main,
        "build_context",
        lambda **kwargs: {
            "schema_version": "1.6-context-quality",
            "prompt": kwargs["prompt"],
            "visual_caption": None,
            "extraction": {"mode": "prompt+detector-fallback", "fallback_used": True},
            "objects": [],
            "attributes": [],
            "relationships": [],
            "scene": {"tags": ["city"]},
            "region_policy": {"confidence_threshold": 0.55, "fallback_region": "full_image"},
        },
    )

    image = Image.new("RGB", (24, 24), (120, 80, 50))
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    with TestClient(main.app) as client:
        response = client.post(
            "/api/v1/images/analyze-upload",
            files={"image": ("custom.png", buffer.getvalue(), "image/png")},
            data={
                "prompt": "a woman holding a red umbrella in a city",
                "generator": "test-generator",
                "category": "realistic",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset_image_name"] == "custom.png"
    assert payload["prompt"].startswith("a woman")
    assert payload["context"]["schema_version"] == "1.6-context-quality"
    assert payload["regions"][0]["is_fallback"] is True
