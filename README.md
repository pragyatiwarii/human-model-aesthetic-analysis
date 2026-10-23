# Human–Model Aesthetic Agreement Analysis

## Modules 1 → 2 integrated build

- **Module 1 — Vennela Laxmi Puli:** AGIQA-3K input, metadata, confidence-filtered YOLO segmentation, and safe `full_image` fallback.
- **Module 2 — Sanyojita Priyadarshani Singh:** controlled region-specific counterfactual generation and Module 3 handoff.

Module 1 remains conceptually frozen at the `v1.4-module-1-vennela` boundary. Module 2 consumes that output rather than replacing its responsibility.

## What Module 2 does

Module 2 takes a Module 1 `image_id` and selected `region_id`, applies one controlled perturbation, and produces three counterfactuals:

```text
Module 1 image + selected region
        ↓
choose one perturbation
        ↓
subtle  |  moderate  |  strong
        ↓
lossless PNG counterfactuals
        ↓
outside-region isolation validation
        ↓
Module 3 handoff JSON
```

Supported perturbations:

- brightness reduction
- contrast reduction
- saturation reduction
- Gaussian blur
- Gaussian noise
- over-sharpening
- colour / hue shift
- pixelation
- structural distortion

Each perturbation has fixed, explicit severity parameters returned in the API response. Gaussian noise also stores its deterministic seed.

## Scientific isolation rule

Counterfactual outputs are saved as **PNG**, not JPEG. JPEG recompression can alter pixels outside the intended edit region even if the image operation itself was masked.

For every generated variant, Module 2 reopens the persisted PNG and validates:

```text
outside_region_max_abs_diff == 0
```

If that invariant fails, the variant is rejected.

The response also records:

- mask source (`segmentation_mask`, `polygon`, `bbox`, or `full_image`)
- region pixel fraction
- inside-region mean absolute difference
- inside-region changed-pixel fraction
- outside-region exact preservation result

## Required local AGIQA-3K data

The project does not redistribute AGIQA-3K. Keep the dataset locally at:

```text
backend/data/agiqa3k/
├── data.csv
└── images/
    ├── AttnGAN_normal_000.jpg
    └── ...
```

If you already have the dataset in the Module 1 project, keep that folder in place when applying the Module 2 patch.

## Recommended environment

- Windows 10/11
- Python 3.11
- Node.js LTS
- NVIDIA GPU optional; CPU is supported

## Backend

From the project root:

```powershell
cd backend
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env -ErrorAction SilentlyContinue
python -m pytest -q
uvicorn app.main:app --reload
```

If `.venv` does not exist yet:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Health check:

```text
http://127.0.0.1:8000/health
```

Expected Module 2 health fields include:

```json
{
  "status": "ok",
  "module": 2,
  "module_1_owner": "Vennela Laxmi Puli",
  "module_2_owner": "Sanyojita Priyadarshani Singh",
  "counterfactual_output": "lossless PNG",
  "perturbation_count": 9
}
```

API docs:

```text
http://127.0.0.1:8000/docs
```

## Frontend

Open a second PowerShell terminal:

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

## How to test Module 2

1. Select an AGIQA-3K sample.
2. Click **Run Module 1**.
3. Select an object region, or use the `full_image` fallback when no reliable object exists.
4. In **Module 2 — Sanyojita**, choose a perturbation such as Gaussian blur.
5. Click **Generate subtle · moderate · strong**.
6. Verify three variants appear.
7. Each variant should show `Outside region: unchanged ✓`.
8. Open the Module 3 handoff JSON.

For a detected object, only the selected region should change. For a Module 1 `full_image` fallback, the whole image is the valid edit region.

## Module 2 API

### List perturbations

```text
GET /api/v1/perturbations
```

### Generate counterfactuals

```text
POST /api/v1/counterfactuals/generate
```

Example request:

```json
{
  "image_id": "<module-1-image-id>",
  "region_id": "<module-1-region-id>",
  "perturbation_type": "blur",
  "severities": ["subtle", "moderate", "strong"]
}
```

### Retrieve a generated batch

```text
GET /api/v1/counterfactuals/batches/{batch_id}
```

### Module 3 handoff

```text
GET /api/v1/counterfactuals/batches/{batch_id}/handoff
```

The handoff contains the original AGIQA identity and human ground truth, the selected Module 1 region, exact perturbation metadata, all three counterfactual URLs, parameters, and isolation validation.

## Git checkpoint

After local testing succeeds:

```powershell
git status
git add .
git commit -m "feat(module-2): add controlled counterfactual generation"
git tag v2.0-module-2-sanyojita
git push origin main
git push origin v2.0-module-2-sanyojita
```

Do not move the existing `v1.4-module-1-vennela` tag. That tag remains the frozen Module 1 checkpoint.
