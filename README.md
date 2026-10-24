# Human–Model Aesthetic Agreement Analysis

## Modules 1 → 4 integrated build — v4.3 context-quality + protocol enhanced


### v4.3 additions for context quality and higher PDF compliance

#### v4.3 context-quality cleanup

Module 1 now separates semantic sources instead of mixing prompt words with detector objects:

- `detector_objects`: YOLO labels with confidence and `high` / `medium` / `uncertain` reliability
- `prompt_entities`: contextual phrases derived from the original prompt
- `vlm_entities`: contextual phrases derived from the VLM caption
- repeated VLM captions and same-head relations such as `rug → with → rug` are removed
- compound attributes such as `black and white ... floor` are attached to the correct head entity
- low-confidence detector-only labels remain usable as editable regions but are explicitly marked for visual verification

This keeps the Module 1 → Module 2 contract backward compatible while making the context panel safer for interpretation and viva discussion.

- **VLM-assisted structured context extraction** using BLIP when available.
- Fail-safe **prompt + YOLO context fallback** if the VLM cannot load/download.
- Structured JSON: objects, attributes, relationships, scene tags, visual caption, extraction source.
- Example relationship output: `woman → holding → red umbrella`, `woman → beside → white car`.
- **Custom image + prompt upload** so a new original image can enter the complete pipeline.
- Exact pairwise human annotation protocol: **Original / Modified / About the same** for subtle/moderate/strong.
- Raw pairwise responses stored in SQLite and propagated to Module 4.
- Preference percentages by severity in the final dashboard.
- Context propagated through Module 1 → 2 → 3 → 4 handoffs and shown in the disagreement report.
- Project-level disagreement aggregation by image category/style in addition to perturbation and region.

The four team module owners remain unchanged; context extraction is an enhancement to **Module 1**, not a fifth ownership module.
This build keeps the completed responsibilities of Modules 1–3 and expands **Module 4 — Pragya Tiwari** so the final quantitative analysis closely follows the supplied B.Tech project specification.

- **Module 1 — Vennela Laxmi Puli:** AGIQA-3K/custom image input, VLM-assisted structured context extraction, metadata, confidence-filtered YOLO segmentation, and safe `full_image` fallback.
- **Module 2 — Sanyojita Priyadarshani Singh:** controlled region-specific counterfactual generation with subtle/moderate/strong severity levels and exact outside-region preservation.
- **Module 3 — Muskan Dubey:** local aesthetic scoring, blind multi-participant 1–5 ratings + Best vote, plus the exact PDF pairwise `Original / Modified / About the same` protocol.
- **Module 4 — Pragya Tiwari:** PDF-aligned human–model quantitative analysis, visualizations, disagreement/error reporting, project dashboard and CSV exports.

## End-to-end workflow

```text
AGIQA-3K OR custom image + prompt
        ↓
Module 1: VLM-assisted structured context + reliable region / full_image fallback
        ↓
Module 2: controlled perturbation
        ↓
subtle | moderate | strong lossless PNG counterfactuals
        ↓
Module 3: model scores + blind ratings/Best + exact pairwise preferences
        ↓
Module 4: quantitative measures + required plots + disagreement report
```

## Module 4 quantitative measures

For every analyzable batch, Module 4 computes:

- Δ model score for subtle/moderate/strong vs original
- human Δ for the same severity levels
- exact Original / Modified / About-the-same preference rates when pairwise annotations exist
- model sensitivity and human sensitivity
- directional human–model agreement rate
- top-choice agreement and human consensus
- Pearson correlation (`r`)
- Spearman rank correlation (`rho`)
- MAE
- RMSE
- candidate-level absolute score gap
- substantial disagreement rate
- rank disagreement
- model severity–response slope
- human severity–response slope
- paired human effect size (Cohen's `dz`)
- 95% confidence intervals for human means/deltas and project top-choice agreement where meaningful

### Substantial disagreement definition

For reproducible charts, a candidate is counted as substantially divergent when:

```text
absolute human–model score gap >= 0.75
```

Both ratings are on the same approximate 1–5 scale. The threshold is included in the API output.

## Module 4 figures/tables

The frontend provides the final project figures requested by the specification:

1. Human vs model sensitivity comparison table
2. Human–model response correlation scatter plot
3. Disagreement-rate chart by perturbation type
4. Disagreement-rate chart by image region
5. Severity–response curve (human vs model)
6. Human–model disagreement heatmap (Region × Perturbation)
7. Summary of statistically/practically important findings
8. Additional disagreement aggregation by image category/style

The dashboard also shows top-choice agreement, directional agreement, MAE/RMSE, correlations, confidence intervals, effect size, score/rank gaps, high-disagreement batches, and warning/error flags.

## Statistical interpretation safeguards

- Correlation is reported as unavailable when one side has no variation.
- Effect size is unavailable when fewer than two paired observations exist or the paired differences have zero standard deviation.
- Human 95% CIs use Student-t critical values for small samples.
- Project top-choice agreement uses a Wilson 95% interval.
- Low participant count is flagged so demo results are not misrepresented as study conclusions.

## Human preference protocol note

The implemented Module 3 now retains the existing blind Best-of-four + 1–5 rating study **and also collects the exact PDF pairwise protocol** for every severity: `Original / Modified / About the same`. Module 4 uses the exact pairwise rates when available and keeps backward compatibility with older batches.

For new v4.2 experiments, Module 4 reports all three exact rates and a severity-level table. Older v4.1 batches remain analyzable with the historical Best-of-four-derived rates.

## Project-level reproducibility

The project dashboard is recomputed from locally stored:

- raw blind human responses
- raw pairwise Original/Modified/About-the-same responses
- model aesthetic scores
- original/counterfactual identifiers
- perturbation type and severity
- region label/fallback status

No graph uses manually entered final-result values.

## Backend setup

```powershell
cd backend
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env -ErrorAction SilentlyContinue
python -m pytest -q
python -m uvicorn app.main:app --reload
```

Expected:

```text
29 passed
```

Health check:

```text
http://127.0.0.1:8000/health
```

The v4.2 health response includes `pdf_quantitative_alignment: true` and lists the implemented statistical families.

## Frontend setup

```powershell
cd frontend
npm.cmd install
Copy-Item .env.example .env -ErrorAction SilentlyContinue
npm.cmd run dev
```

Open:

```text
http://127.0.0.1:5173
```

## Module 4 API

Single-batch analysis:

```text
GET /api/v1/module4/batches/{batch_id}/analysis
```

Project dashboard / all figure data:

```text
GET /api/v1/module4/dashboard
```

Single-batch candidate CSV:

```text
GET /api/v1/module4/batches/{batch_id}/report.csv
```

Project quantitative CSV:

```text
GET /api/v1/module4/dashboard/report.csv
```

## How to test the final module

1. Complete one Modules 1→3 experiment.
2. Collect at least 2–3 participant responses for a meaningful software test.
3. In Module 4 click **Analyze this human–model batch**.
4. Verify the batch metrics, severity curve, sensitivity/effect-size table and disagreement report.
5. Click **Refresh project dashboard**.
6. Verify the correlation plot, perturbation disagreement chart, region disagreement chart, severity table and Region × Perturbation heatmap.
7. Export the batch CSV and project CSV.
8. Add multiple perturbations/regions over time; the project figures automatically become richer.

## Remaining specification gap

The largest remaining requirement outside Module 4 is richer VLM/MLLM context extraction (objects + attributes + relationships + scene context). The current Module 1 intentionally focuses on prompt/AGIQA metadata plus YOLO region intelligence. The final quantitative/statistical portion is now much closer to the supplied PDF, but the project should not claim a full VLM relationship graph unless that feature is actually added.

## Checkpoints

```text
v1.4-module-1-vennela
        ↓
v2.0-module-2-sanyojita
        ↓
v3.0-module-3-muskan
        ↓
v4.3-module-4-pragya-context-quality   (create only after local UI validation)
```
