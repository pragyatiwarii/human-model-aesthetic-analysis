from PIL import Image

from app.services.context import build_context, prompt_colors


def test_prompt_colors():
    result = prompt_colors("a red umbrella beside a white car")
    assert "red" in result
    assert "white" in result


def test_context_has_no_caption_dependency():
    image = Image.new("RGB", (32, 32), (100, 100, 100))

    context = build_context(
        image=image,
        prompt="a cosmic universe",
        regions=[
            {
                "label": "full_image",
                "confidence": 1.0,
            }
        ],
        confidence_threshold=0.55,
    )

    assert context["prompt"] == "a cosmic universe"
    assert "caption" not in context
    assert context["objects"] == []
    assert context["region_policy"]["fallback_region"] == "full_image"
