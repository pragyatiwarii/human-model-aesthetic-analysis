from PIL import Image

from app.services.context import build_context, prompt_colors


def test_prompt_colors():
    result = prompt_colors("a red umbrella beside a white car")
    assert "red" in result
    assert "white" in result


def test_context_fallback_is_structured_and_safe():
    image = Image.new("RGB", (32, 32), (100, 100, 100))

    context = build_context(
        image=image,
        prompt="a cosmic universe",
        regions=[{"label": "full_image", "confidence": 1.0}],
        confidence_threshold=0.55,
        enable_vlm=False,
    )

    assert context["prompt"] == "a cosmic universe"
    assert context["schema_version"] == "1.6-context-quality"
    assert context["visual_caption"] is None
    assert context["extraction"]["fallback_used"] is True
    assert context["region_policy"]["fallback_region"] == "full_image"
    assert "universe" in context["scene"]["tags"]


def test_context_extracts_prompt_attributes_relationships_and_detector_objects():
    image = Image.new("RGB", (32, 32), (120, 120, 120))
    context = build_context(
        image=image,
        prompt="a woman holding a red umbrella beside a white car in a rainy city",
        regions=[
            {"label": "person", "confidence": 0.92},
            {"label": "car", "confidence": 0.87},
        ],
        confidence_threshold=0.55,
        enable_vlm=False,
        style="realistic",
    )

    object_labels = {item["label"] for item in context["objects"]}
    assert "person" in object_labels
    assert "car" in object_labels
    attrs = {(item["object"], item["value"]) for item in context["attributes"]}
    assert ("umbrella", "red") in attrs
    assert ("car", "white") in attrs
    assert any(item["relation"] == "holding" for item in context["relationships"])
    assert "city" in context["scene"]["tags"]
    assert "rainy" in context["scene"]["tags"]


def test_vlm_caption_override_is_incorporated_without_loading_model():
    image = Image.new("RGB", (32, 32), (120, 120, 120))
    context = build_context(
        image=image,
        prompt="a person near a car",
        regions=[{"label": "person", "confidence": 0.9}],
        confidence_threshold=0.55,
        enable_vlm=True,
        caption_override="a person standing near a blue car on a street",
    )
    assert context["visual_caption"].startswith("a person")
    assert context["extraction"]["mode"] == "vlm+prompt+detector"
    assert any(item["source"] == "vlm" for item in context["relationships"])
    assert "street" in context["scene"]["tags"]


def test_context_quality_separates_detector_objects_from_prompt_entities_and_cleans_duplicates():
    image = Image.new("RGB", (32, 32), (120, 120, 120))
    context = build_context(
        image=image,
        prompt="a living room with a black and white chevron floor, elevation view, blurred detail",
        regions=[{"label": "laptop", "confidence": 0.631}],
        confidence_threshold=0.55,
        enable_vlm=True,
        caption_override="a white and black striped rug with a black and white striped rug",
    )

    assert context["schema_version"] == "1.6-context-quality"
    assert [item["label"] for item in context["objects"]] == ["laptop"]
    assert context["detector_objects"][0]["label"] == "laptop"
    assert context["detector_objects"][0]["reliability"] == "uncertain"
    assert context["visual_caption"] == "a white and black striped rug"

    prompt_labels = {item["label"] for item in context["prompt_entities"]}
    assert "living room" in prompt_labels
    assert all(label not in {"living", "striped", "chevron"} for label in prompt_labels)

    attrs = {(item["object"], item["value"]) for item in context["attributes"]}
    assert ("floor", "black and white") in attrs
    assert ("image", "elevation") in attrs
    assert ("image", "blurred") in attrs

    assert not any(
        item["relation"] == "with" and item["subject"].endswith("rug") and item["object"].endswith("rug")
        for item in context["relationships"]
    )
