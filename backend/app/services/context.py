from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from PIL import Image


COLOR_WORDS = {
    "red", "blue", "green", "yellow", "black", "white", "orange",
    "purple", "pink", "brown", "gray", "grey", "gold", "silver",
}

SCENE_WORDS = {
    "indoor", "outdoor", "city", "street", "road", "beach", "forest",
    "mountain", "room", "kitchen", "office", "park", "rain", "rainy",
    "snow", "snowy", "night", "day", "sunset", "sunrise", "space",
    "universe", "ocean", "sea", "desert", "field", "studio", "portrait",
}

RELATION_MARKERS = (
    "in front of", "next to", "holding", "carrying", "wearing", "riding",
    "beside", "behind", "under", "above", "near", "with", "on",
)

_STOPWORDS = {
    "a", "an", "the", "of", "in", "at", "to", "for", "and", "or",
    "that", "which", "who", "this", "these", "those", "looks", "look",
    "style", "image", "photo", "picture", "scene", "view", "realistic",
    "abstract", "beautiful", "detailed", "high", "low", "normal",
}

_DESCRIPTOR_WORDS = {
    "striped", "chevron", "blurred", "blurry", "detailed", "realistic",
    "abstract", "elevation", "top", "side", "front", "rear", "hyper",
    "beautiful", "dark", "bright", "small", "large", "big", "tiny",
}

_NON_ENTITY_HEADS = {
    "detail", "view", "style", "image", "photo", "picture", "scene",
}

_VLM_STATE: dict[str, Any] = {
    "processor": None,
    "model": None,
    "device": None,
    "model_name": None,
    "error": None,
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

    output: list[dict[str, Any]] = []
    for rgb, count in counts.most_common(k):
        triplet = tuple(int(v) for v in rgb)
        output.append({
            "rgb": list(triplet),
            "name": _simple_color(triplet),
            "share": round(count / total, 4),
        })
    return output


def _clean_phrase(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\- ]+", " ", text.lower())
    tokens = [t for t in text.split() if t not in {"a", "an", "the"}]
    return " ".join(tokens).strip()


def _head_noun(phrase: str) -> str:
    words = [w for w in _clean_phrase(phrase).split() if w not in COLOR_WORDS and w not in _DESCRIPTOR_WORDS]
    return words[-1] if words else ""


def _normalise_entity_phrase(text: str) -> str:
    phrase = _clean_phrase(text)
    phrase = re.sub(r"^(?:and|or)\s+", "", phrase).strip()
    phrase = re.sub(r"\s+", " ", phrase)
    return phrase


def _clean_caption(text: str | None) -> str | None:
    if not text:
        return None
    caption = re.sub(r"\s+", " ", text.strip())
    # BLIP occasionally emits "a X with a X". Keep one description when both sides
    # clearly refer to the same head noun.
    match = re.match(r"^(.*?)(?:\s+with\s+)(.*)$", caption, flags=re.I)
    if match:
        left = _normalise_entity_phrase(match.group(1))
        right = _normalise_entity_phrase(match.group(2))
        if left and right and _head_noun(left) and _head_noun(left) == _head_noun(right):
            caption = match.group(1).strip().rstrip(" ,;.")
    return caption[:1].lower() + caption[1:] if caption else None


def _extract_relationships(text: str, source: str) -> list[dict[str, str]]:
    """Extract conservative subject–relation–object triples without an NLP dependency."""
    raw = re.sub(r"[,;]", " . ", text.lower())
    relation_pattern = re.compile(
        r"\b(" + "|".join(re.escape(marker) for marker in sorted(RELATION_MARKERS, key=len, reverse=True)) + r")\b"
    )
    triples: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for clause in re.split(r"[.!?]", raw):
        clause = clause.strip()
        if not clause:
            continue
        matches = list(relation_pattern.finditer(clause))
        if not matches:
            continue
        subject_words = _normalise_entity_phrase(clause[: matches[0].start()]).split()
        base_subject = " ".join(subject_words[-5:])
        if not base_subject or _head_noun(base_subject) in _NON_ENTITY_HEADS:
            continue
        for index, match in enumerate(matches):
            marker = match.group(1)
            end = matches[index + 1].start() if index + 1 < len(matches) else len(clause)
            obj = _normalise_entity_phrase(clause[match.end() : end])
            obj = re.split(r"\s+(?:in|at)\s+", obj, maxsplit=1)[0].strip()
            obj = " ".join(obj.split()[:7])
            if not obj or _head_noun(obj) in _NON_ENTITY_HEADS:
                continue
            # "rug with rug" and similar VLM repetitions are not meaningful relations.
            if marker == "with" and _head_noun(base_subject) == _head_noun(obj):
                continue
            key = (base_subject, marker.replace(" ", "_"), obj)
            if key in seen:
                continue
            seen.add(key)
            triples.append({
                "subject": base_subject,
                "relation": marker.replace(" ", "_"),
                "object": obj,
                "source": source,
            })
    return triples


def _prompt_attributes(prompt: str) -> list[dict[str, str]]:
    attrs: list[dict[str, str]] = []
    lower = prompt.lower()
    color_pattern = "|".join(sorted(COLOR_WORDS, key=len, reverse=True))

    consumed: list[tuple[int, int]] = []
    # Handle compound colours first: "black and white chevron floor" -> floor: black and white.
    compound = re.compile(rf"\b({color_pattern})\s+and\s+({color_pattern})\s+((?:[a-z-]+\s+){{0,2}}[a-z-]+)")
    for match in compound.finditer(lower):
        phrase = match.group(3).strip()
        head = _head_noun(phrase) or phrase.split()[-1]
        attrs.append({
            "object": head,
            "attribute_type": "color",
            "value": f"{match.group(1)} and {match.group(2)}",
            "source": "prompt",
        })
        consumed.append(match.span())

    def covered(span):
        return any(span[0] >= a and span[1] <= b for a, b in consumed)

    for match in re.finditer(rf"\b({color_pattern})\s+([a-z][a-z-]*)", lower):
        if covered(match.span()) or match.group(2) in {"and", "or"}:
            continue
        attrs.append({
            "object": match.group(2),
            "attribute_type": "color",
            "value": match.group(1),
            "source": "prompt",
        })

    descriptor_patterns = {
        "blurred": ("image", "detail", "blurred"),
        "blurry": ("image", "detail", "blurred"),
        "elevation view": ("image", "viewpoint", "elevation"),
        "top view": ("image", "viewpoint", "top"),
        "side view": ("image", "viewpoint", "side"),
        "front view": ("image", "viewpoint", "front"),
    }
    for phrase, (obj, typ, value) in descriptor_patterns.items():
        if phrase in lower:
            attrs.append({"object": obj, "attribute_type": typ, "value": value, "source": "prompt"})

    # Stable de-duplication.
    unique = []
    seen = set()
    for item in attrs:
        key = (item["object"], item["attribute_type"], item["value"], item["source"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _contextual_entities(text: str, source: str) -> list[dict[str, str]]:
    """Return contextual phrases without pretending they are detector-confirmed objects."""
    entities: list[dict[str, str]] = []
    seen: set[str] = set()

    # Relation subjects and objects are the highest-value phrases.
    for rel in _extract_relationships(text, source):
        for phrase in (rel["subject"], rel["object"]):
            phrase = _normalise_entity_phrase(phrase)
            if not phrase or phrase in seen:
                continue
            seen.add(phrase)
            entities.append({"label": phrase, "source": source, "kind": "context_phrase"})

    # Add simple determiner noun phrases only when no relation extracted them.
    phrase_pattern = re.compile(r"\b(?:a|an|the)\s+([a-z-]+(?:\s+(?!with\b|in\b|on\b|near\b|beside\b|behind\b|under\b|above\b|holding\b|carrying\b|wearing\b|riding\b)[a-z-]+){0,5})")
    for match in phrase_pattern.finditer(text.lower()):
        phrase = _normalise_entity_phrase(match.group(1))
        words = phrase.split()
        head = _head_noun(phrase)
        if not phrase or not head or head in _NON_ENTITY_HEADS:
            continue
        # Avoid pure descriptor fragments (e.g. "blurred detail").
        if all(w in _DESCRIPTOR_WORDS or w in COLOR_WORDS for w in words):
            continue
        if phrase not in seen:
            seen.add(phrase)
            entities.append({"label": phrase, "source": source, "kind": "context_phrase"})
    return entities


def _detection_reliability(label: str, confidence: float, prompt: str, caption: str | None) -> dict[str, Any]:
    evidence_text = f"{prompt} {caption or ''}".lower()
    mentioned = bool(re.search(rf"\b{re.escape(label.lower())}\b", evidence_text))
    if confidence >= 0.75:
        level = "high"
    elif confidence >= 0.65 and mentioned:
        level = "medium"
    else:
        level = "uncertain"
    return {
        "level": level,
        "mentioned_in_prompt_or_caption": mentioned,
        "note": (
            "Detector-supported object." if level == "high"
            else "Detector result has supporting text evidence." if level == "medium"
            else "Detector-only result; verify visually before treating as semantic ground truth."
        ),
    }


def _scene_tags(prompt: str, caption: str | None) -> list[str]:
    words = set(re.findall(r"[a-z]+", f"{prompt} {caption or ''}".lower()))
    return sorted(words.intersection(SCENE_WORDS))


def _load_blip(model_name: str, device_preference: str) -> tuple[Any, Any, str]:
    if _VLM_STATE["model"] is not None and _VLM_STATE["model_name"] == model_name:
        return _VLM_STATE["processor"], _VLM_STATE["model"], _VLM_STATE["device"]
    if _VLM_STATE["error"] and _VLM_STATE["model_name"] == model_name:
        raise RuntimeError(_VLM_STATE["error"])

    try:
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        if device_preference == "cpu":
            device = "cpu"
        elif device_preference == "cuda":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name)
        model = model.to(device)
        model.eval()
        _VLM_STATE.update({
            "processor": processor,
            "model": model,
            "device": device,
            "model_name": model_name,
            "error": None,
        })
        return processor, model, device
    except Exception as exc:  # fail-safe by design
        message = f"{type(exc).__name__}: {exc}"
        _VLM_STATE.update({
            "processor": None,
            "model": None,
            "device": None,
            "model_name": model_name,
            "error": message,
        })
        raise RuntimeError(message) from exc


def generate_vlm_caption(
    image: Image.Image,
    *,
    enabled: bool,
    model_name: str,
    device_preference: str = "auto",
    max_new_tokens: int = 40,
) -> tuple[str | None, dict[str, Any]]:
    status = {
        "enabled": enabled,
        "model": model_name,
        "used": False,
        "device": None,
        "error": None,
    }
    if not enabled:
        status["error"] = "VLM context extraction disabled by configuration."
        return None, status

    try:
        processor, model, device = _load_blip(model_name, device_preference)
        import torch

        inputs = processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens)
        caption = processor.decode(output[0], skip_special_tokens=True).strip()
        status.update({"used": bool(caption), "device": device})
        return caption or None, status
    except Exception as exc:
        status["error"] = str(exc)
        return None, status


def context_vlm_status(*, enabled: bool, model_name: str) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "model": model_name,
        "loaded": _VLM_STATE["model"] is not None,
        "device": _VLM_STATE["device"],
        "last_error": _VLM_STATE["error"],
        "fallback_available": True,
    }


def build_context(
    image: Image.Image,
    prompt: str,
    regions: list[dict[str, Any]],
    confidence_threshold: float,
    *,
    adj1: str | None = None,
    adj2: str | None = None,
    style: str | None = None,
    enable_vlm: bool = False,
    vlm_model: str = "Salesforce/blip-image-captioning-base",
    vlm_device: str = "auto",
    vlm_max_new_tokens: int = 40,
    caption_override: str | None = None,
) -> dict[str, Any]:
    caption: str | None
    if caption_override is not None:
        caption = caption_override.strip() or None
        vlm_status = {
            "enabled": True,
            "model": "test-override",
            "used": bool(caption),
            "device": "test",
            "error": None,
        }
    else:
        caption, vlm_status = generate_vlm_caption(
            image,
            enabled=enable_vlm,
            model_name=vlm_model,
            device_preference=vlm_device,
            max_new_tokens=vlm_max_new_tokens,
        )

    caption = _clean_caption(caption)

    detector_objects: list[dict[str, Any]] = []
    detector_counts: Counter[str] = Counter()
    for region in regions:
        label = region.get("label")
        if not label or label == "full_image":
            continue
        confidence = float(region.get("confidence", 0.0))
        detector_counts[label] += 1
        reliability = _detection_reliability(label, confidence, prompt, caption)
        detector_objects.append({
            "label": label,
            "confidence": round(confidence, 6),
            "reliability": reliability["level"],
            "text_supported": reliability["mentioned_in_prompt_or_caption"],
            "note": reliability["note"],
            "source": "detector",
        })

    prompt_entities = _contextual_entities(prompt, "prompt")
    vlm_entities = _contextual_entities(caption, "vlm") if caption else []

    relationships = _extract_relationships(prompt, "prompt")
    if caption:
        existing = {(r["subject"], r["relation"], r["object"]) for r in relationships}
        for triple in _extract_relationships(caption, "vlm"):
            key = (triple["subject"], triple["relation"], triple["object"])
            if key not in existing:
                relationships.append(triple)
                existing.add(key)

    attributes = _prompt_attributes(prompt)
    for name, value in (("adj1", adj1), ("adj2", adj2), ("style", style)):
        if value:
            attributes.append({
                "object": "image",
                "attribute_type": name,
                "value": str(value),
                "source": "AGIQA-3K",
            })

    mode = "vlm+prompt+detector" if vlm_status["used"] else "prompt+detector-fallback"
    return {
        "schema_version": "1.6-context-quality",
        "prompt": prompt,
        "visual_caption": caption,
        "extraction": {
            "mode": mode,
            "vlm": vlm_status,
            "fallback_used": not bool(vlm_status["used"]),
        },
        # Backward-compatible "objects" now means detector-confirmed candidates only.
        "objects": [
            {
                "label": label,
                "count": int(detector_counts[label]),
                "sources": ["detector"],
            }
            for label in sorted(detector_counts, key=lambda item: (-detector_counts[item], item))
        ],
        "detector_objects": detector_objects,
        "prompt_entities": prompt_entities,
        "vlm_entities": vlm_entities,
        "attributes": attributes,
        "relationships": relationships,
        "scene": {
            "tags": _scene_tags(prompt, caption),
            "style": style,
            "adjectives": [value for value in (adj1, adj2) if value],
            "prompt_colors": prompt_colors(prompt),
            "dominant_colors": dominant_colors(image),
        },
        "region_policy": {
            "confidence_threshold": confidence_threshold,
            "fallback_region": "full_image",
            "low_confidence_detections_rejected": True,
        },
    }
