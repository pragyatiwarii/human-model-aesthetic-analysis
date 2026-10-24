# Context + PDF protocol enhancement — v4.3

This enhancement closes the largest remaining gaps in the supplied B.Tech specification without changing the four team ownership modules.

## 1. Module 1 structured context extraction

Module 1 now produces `context.schema_version = 1.6-context-quality` containing:

- prompt
- optional BLIP visual caption
- object list with detector/prompt/VLM sources
- explicit attributes, including colour and AGIQA style/adjectives
- subject–relation–object relationships
- scene/context tags
- dominant image colours
- extraction mode and VLM/fallback status

The default VLM is `Salesforce/blip-image-captioning-base`. It is loaded lazily on the first Module 1 analysis. If the model cannot be downloaded or loaded, Module 1 continues with prompt + YOLO structured extraction; Modules 2–4 are not blocked.

Example:

```json
{
  "relationships": [
    {"subject": "woman", "relation": "holding", "object": "red umbrella", "source": "prompt"},
    {"subject": "woman", "relation": "beside", "object": "white car", "source": "prompt"}
  ]
}
```

## 2. New original image input

A user can upload a JPG/PNG/WEBP image (up to 20 MB), supply its original generation prompt, generator/source and optional category/style, and run the same Module 1 → 4 workflow.

AGIQA-3K remains the main research dataset and is still used for the trained aesthetic regressor.

## 3. Exact human preference protocol

Module 3 keeps the existing blind A/B/C/D 1–5 rating + Best vote study. After that response, the same participant also completes three pairwise questions:

- Original vs Subtle
- Original vs Moderate
- Original vs Strong

For each pair the participant selects exactly one:

- Original
- Modified
- About the same

Raw pairwise responses are stored in SQLite and carried into Module 4.

## 4. Module 4 use

When pairwise responses exist, Module 4 computes the exact original/modified/same percentages from those raw responses, including rates by severity. Older batches without pairwise responses remain analyzable using the previous Best-of-four derived rates.

Structured context is also included in the Module 4 handoff and displayed with disagreement/error cases. The project dashboard now additionally aggregates disagreement by image category/style.

## Environment additions

The backend requirements now include `transformers` and `python-multipart`.

Relevant `.env` controls:

```text
ENABLE_CONTEXT_VLM=true
CONTEXT_VLM_MODEL=Salesforce/blip-image-captioning-base
CONTEXT_VLM_DEVICE=auto
CONTEXT_VLM_MAX_NEW_TOKENS=40
```

For a faster/offline demonstration, set `ENABLE_CONTEXT_VLM=false`; structured prompt + detector fallback still works.

## Validation

Backend automated tests: **29 passed**.


## v4.3 semantic cleanup

The context payload now separates detector-confirmed candidates from prompt/VLM contextual phrases. Detector entries include confidence, textual support, and a reliability label. Repeated captions and low-value self-relations are removed before handoff.
