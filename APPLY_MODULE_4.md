# Apply Module 4 v4.2 — PDF-aligned final analysis

This patch upgrades the final Pragya module without changing the responsibilities of Modules 1–3.

## What v4.2 adds

- Δ model score and human Δ per severity
- derived human original-vs-counterfactual preference rates
- model sensitivity and human sensitivity
- directional human–model agreement rate
- Pearson r and Spearman rho
- MAE and RMSE
- substantial disagreement rate (score gap >= 0.75 on the 1–5 scale)
- severity–response slopes for model and humans
- paired human effect size (Cohen's dz)
- 95% confidence intervals for human means/deltas and overall top-choice agreement
- human-vs-model sensitivity table
- correlation scatter plot
- disagreement-rate chart by perturbation
- disagreement-rate chart by image region
- severity–response curve
- Region × Perturbation disagreement heatmap
- automatically generated findings summary
- per-batch and project-level CSV exports

## Apply to your current Modules 1→3 project

Copy the patch contents into the Git-tracked project root and replace matching files.

Do not copy or commit:

- `backend/.venv/`
- `backend/data/agiqa3k/images/`
- `backend/data/runtime/`
- `frontend/node_modules/`
- `.env`

## Validate

```powershell
cd backend
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python -m uvicorn app.main:app --reload
```

Expected backend tests:

```text
24 passed
```

Then in a second terminal:

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

Open `http://127.0.0.1:5173`, complete one Modules 1→3 batch, then run Module 4 analysis and refresh the project dashboard.

Do not create the final Git tag until the UI and charts have been checked locally.


For the context/custom-input/pairwise additions, also read `APPLY_CONTEXT_PROTOCOL_ENHANCEMENT.md`.
