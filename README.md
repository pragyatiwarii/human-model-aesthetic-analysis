# Human–Model Aesthetic Agreement Analysis
## Module 1 — Vennela Laxmi Puli — clean v1.4

This is a clean rebuild of Module 1. It intentionally contains only the
information required by later modules.

## Module 1 does

- browse the AGIQA-3K dataset
- load the original AGIQA-3K image and prompt
- retain AGIQA-3K MOS/STD metadata
- detect/segment reliable editable regions using YOLO segmentation
- reject weak detections below the configured confidence threshold
- return a safe `full_image` fallback if no reliable object is detected
- save region polygons/masks when segmentation is available
- provide a stable Module 2 handoff JSON

## Module 1 does NOT do

- BLIP caption generation
- generic image captioning
- counterfactual image modification
- aesthetic scoring
- human voting
- human/model statistical comparison

Those are not needed for the Module 1 → Module 2 boundary.

## Required local dataset

The ZIP does not redistribute AGIQA-3K. Put your existing dataset here:

```text
backend/data/agiqa3k/
├── data.csv
└── images/
    ├── AttnGAN_normal_000.jpg
    └── ...
```

If you already have a working AGIQA-3K folder from the earlier version,
copy the entire `agiqa3k` folder into `backend/data/`.

## Recommended environment

- Windows 10/11
- Python 3.11
- Node.js LTS
- NVIDIA GPU optional; CPU is supported

## Backend

```powershell
cd backend

py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

Copy-Item .env.example .env

uvicorn app.main:app --reload
```

Check:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/api/v1/dataset/status
http://127.0.0.1:8000/docs
```

Expected dataset status after copying the complete dataset:

```json
{
  "dataset": "AGIQA-3K",
  "metadata_found": true,
  "record_count": 2982,
  "available_image_count": 2982,
  "missing_image_count": 0,
  "ready": true
}
```

## Frontend

Open a second terminal:

```powershell
cd frontend
npm.cmd install
Copy-Item .env.example .env
npm.cmd run dev
```

Open:

```text
http://127.0.0.1:5173
```

## Core workflow

```text
AGIQA-3K sample
      ↓
original image + prompt + MOS metadata
      ↓
YOLO segmentation/detection
      ↓
confidence filtering
      ↓
reliable regions
      ↓
or full_image fallback
      ↓
region ID + bbox + polygon/mask
      ↓
Module 2 handoff JSON
```

## Detection confidence

The default is:

```env
DETECTION_CONFIDENCE=0.55
```

A weak prediction such as `bear = 0.36` will therefore be rejected.

## Module 2 handoff

After running Module 1:

```http
GET /api/v1/images/{image_id}/handoff
```

This is the exact boundary for Sanyojita's Module 2.

## Tests

```powershell
cd backend
pytest -q
```

## Git checkpoint

After testing locally:

```bash
git add .
git commit -m "feat(module-1): clean AGIQA region handoff pipeline"
git tag v1.4-module-1-vennela
git push origin main
git push origin v1.4-module-1-vennela
```

# MODULE 1 ENDS HERE — SAFE TO PUSH AFTER LOCAL TESTING
