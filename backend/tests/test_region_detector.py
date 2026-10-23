from app.services.region_detector import RegionDetector


def test_fallback_region():
    region = RegionDetector.fallback("abc")

    assert region["region_id"] == "abc-r001"
    assert region["label"] == "full_image"
    assert region["confidence"] == 1.0
    assert region["bbox"] == [0.0, 0.0, 1.0, 1.0]
    assert region["is_fallback"] is True
