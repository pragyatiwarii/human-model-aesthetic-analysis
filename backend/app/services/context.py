from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np
from PIL import Image


COLOR_WORDS = {
    "red",
    "blue",
    "green",
    "yellow",
    "black",
    "white",
    "orange",
    "purple",
    "pink",
    "brown",
    "gray",
    "grey",
    "gold",
    "silver",
}


def prompt_colors(prompt: str) -> list[str]:
    words = set(re.findall(r"[a-z]+", prompt.lower()))
    return sorted(words.intersection(COLOR_WORDS))


def _simple_color(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb

    if max(rgb) < 45:
        return "black"
    if min(rgb) > 215:
        return "white"
    if max(rgb) - min(rgb) < 22:
        return "gray"
    if r > g * 1.25 and r > b * 1.25:
        return "red"
    if g > r * 1.20 and g > b * 1.15:
        return "green"
    if b > r * 1.20 and b > g * 1.10:
        return "blue"
    if r > 150 and g > 120 and b < 100:
        return "yellow/orange"
    if r > 120 and b > 110 and g < 120:
        return "purple/pink"
    return "mixed"


def dominant_colors(image: Image.Image, k: int = 4) -> list[dict[str, Any]]:
    thumb = image.convert("RGB").resize((96, 96))
    pixels = np.asarray(thumb).reshape(-1, 3)

    quantized = (pixels // 32) * 32 + 16
    counts = Counter(map(tuple, quantized.tolist()))
    total = len(pixels)

    output = []
    for rgb, count in counts.most_common(k):
        triplet = tuple(int(v) for v in rgb)
        output.append(
            {
                "rgb": list(triplet),
                "name": _simple_color(triplet),
                "share": round(count / total, 4),
            }
        )

    return output


def build_context(
    image: Image.Image,
    prompt: str,
    regions: list[dict[str, Any]],
    confidence_threshold: float,
) -> dict[str, Any]:
    labels = [
        r["label"]
        for r in regions
        if r["label"] != "full_image"
    ]

    counts = Counter(labels)

    return {
        "prompt": prompt,
        "objects": [
            {"label": label, "count": count}
            for label, count in counts.most_common()
        ],
        "attributes": {
            "prompt_colors": prompt_colors(prompt),
            "dominant_colors": dominant_colors(image),
        },
        "region_policy": {
            "confidence_threshold": confidence_threshold,
            "fallback_region": "full_image",
            "low_confidence_detections_rejected": True,
        },
    }
