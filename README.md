# Human-Model Aesthetic Agreement Analysis

## Modules 1 -> 3 integrated build

- **Module 1 - Vennela Laxmi Puli:** AGIQA-3K input, metadata, confidence-filtered YOLO segmentation, safe `full_image` fallback.
- **Module 2 - Sanyojita Priyadarshani Singh:** controlled region-specific counterfactual generation with subtle/moderate/strong variants and exact outside-region preservation.
- **Module 3 - Muskan Dubey:** model aesthetic scoring plus blind human aesthetic ratings/preferences and a Module 4 analysis handoff.

Module 1 remains frozen at `v1.4-module-1-vennela`. Module 2 remains frozen at `v2.0-module-2-sanyojita`.

## Module 3 workflow

```text
Module 2 batch
    |
    +-- original
    +-- subtle
    +-- moderate
    +-- strong
            |
            v
AGIQA-trained local aesthetic regressor
            |
            +--> model score for all 4 candidates
            |
            v
blind human participant view
            |
            +--> 1-5 rating for every candidate
            +--> one preferred candidate
            |
            v
Module 4 handoff
```

## Why the Module 3 scorer is local

The default scorer is deliberately reproducible and does not require downloading a large third-party foundation model at demo time. It trains a small ridge-regression model from the project's own AGIQA-3K `mos_quality` values and deterministic image features.

This should be described accurately in the report as a **project-specific AGIQA-trained aesthetic regressor**, not as a general-purpose aesthetic foundation model.

The model reports held-out validation metrics so its limitations are visible.

### Features used

- luminance mean and spread,
- saturation mean and spread,
- colorfulness,
- gradient energy,
- edge density,
- luminance entropy,
- dark/bright clipping fractions,
- centre-vs-global luminance balance,
- horizontal symmetry.

Default training limit:

```text
AESTHETIC_TRAIN_LIMIT=1000
```

Set it to `0` to use all locally available AGIQA-3K images.

The trained model is cached at:

```text
backend/data/runtime/models/aesthetic_ridge_v1.json
```

Runtime files are excluded by `.gitignore`.

## Human preference design

A participant judges four candidates from one Module 2 experiment:

- original,
- subtle,
- moderate,
- strong.

For every candidate the participant gives a 1-5 aesthetic rating and also chooses one overall Best candidate.

The UI includes a **blind participant view**. Before submission it hides:

- AGIQA MOS values,
- original/counterfactual identity,
- severity identity,
- perturbation identity,
- model scores.

After submission, identities and model scores can be revealed for researcher inspection.

## Required local AGIQA-3K data

```text
backend/data/agiqa3k/
|-- data.csv
`-- images/
    |-- AttnGAN_normal_000.jpg
    `-- ...
```

The repository does not redistribute the dataset.

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

If `.venv` does not exist:

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

Module 3 health information includes the model training status.

API documentation:

```text
http://127.0.0.1:8000/docs
```

## Frontend

Open a second PowerShell:

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

## Module 3 API

### Model status

```text
GET /api/v1/aesthetic-model/status
```

### Train/cache local aesthetic model

```text
POST /api/v1/aesthetic-model/train
```

Body:

```json
{"force": false}
```

### Score one Module 2 batch

```text
POST /api/v1/aesthetic/batches/{batch_id}/score
```

Returns a model score for original + subtle + moderate + strong.

### Save human preference

```text
POST /api/v1/human-preferences
```

The request contains participant code, one 1-5 rating per candidate and one preferred candidate.

### Module 4 handoff

```text
GET /api/v1/module3/batches/{batch_id}/handoff
```

The handoff is available after the batch has model scores and at least one human preference response.

## Validation completed before packaging

```text
13 backend tests passed
```

The aesthetic model was also smoke-tested for train -> cache -> load -> score behavior.

The package installer was not available reliably in the build sandbox, so the final Vite runtime check must be performed on the Windows laptop using `npm.cmd install` and `npm.cmd run dev` before GitHub push.

## Git checkpoint after local validation

```powershell
git add .
git commit -m "feat(module-3): add aesthetic scoring and human preferences"
git tag v3.0-module-3-muskan
git push origin main
git push origin v3.0-module-3-muskan
```
