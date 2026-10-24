# Module 3 checkpoint - Muskan Dubey

## Responsibility

Module 3 performs **aesthetic scoring and human preference collection** on the controlled image sets produced by Module 2.

It does not modify the Module 1 region detector or the Module 2 perturbation engine.

## Input boundary

Module 3 consumes one Module 2 batch containing:

- the original AGIQA-3K image,
- the selected Module 1 region,
- one perturbation type,
- subtle, moderate and strong lossless counterfactuals,
- exact perturbation parameters and isolation-validation metadata.

## Aesthetic model

Module 3 trains a local project-specific ridge regressor on AGIQA-3K `mos_quality`.

The default training subset is deterministic and capped at 1000 available AGIQA-3K images for practical laptop runtime. Set `AESTHETIC_TRAIN_LIMIT=0` to use every available image.

The model uses deterministic visual features including luminance, saturation, colorfulness, gradient/edge information, entropy, clipping fractions, centre luminance balance and horizontal symmetry.

The trained model is cached under `backend/data/runtime/models/` and is ignored by Git.

The UI reports validation MAE, RMSE and R-squared so the model's limitations are visible rather than hidden.

## Human preference collection

For each experiment, humans judge exactly four candidates:

1. original,
2. subtle counterfactual,
3. moderate counterfactual,
4. strong counterfactual.

Each participant provides:

- a 1-5 aesthetic rating for every candidate,
- one preferred/best candidate,
- a participant code.

The **blind participant view** hides upstream MOS values, perturbation identities, candidate identities and model scores until the response is submitted.

## Module 4 handoff

The Module 4 handoff contains:

- original AGIQA metadata,
- Module 1 region information,
- Module 2 perturbation information,
- model version and validation information,
- model aesthetic score for all four candidates,
- all human rating/preference records,
- aggregate average human rating per candidate,
- preferred-candidate counts.

Endpoint:

```text
GET /api/v1/module3/batches/{batch_id}/handoff
```

## Validation

Backend automated tests at packaging time:

```text
13 passed
```

A separate model smoke test verified training, caching, loading and scoring on a synthetic AGIQA-style dataset.

## Git checkpoint

After local laptop UI testing succeeds:

```powershell
git add .
git commit -m "feat(module-3): add aesthetic scoring and human preferences"
git tag v3.0-module-3-muskan
git push origin main
git push origin v3.0-module-3-muskan
```
