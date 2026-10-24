# Module 4 checkpoint — Pragya Tiwari — v4.2 context + protocol enhanced

## Responsibility

Module 4 performs the final reproducible human–model quantitative analysis over stored Module 3 model scores, raw human annotations and Module 2 perturbation metadata.

## Quantitative measures

- model score change (Δ)
- human response change (Δ)
- human preference rate derived from blind Best-of-four votes
- model sensitivity
- human sensitivity
- directional human–model agreement rate
- Pearson correlation
- Spearman rank correlation
- MAE
- RMSE
- substantial disagreement rate
- severity–response slope
- paired human effect size (Cohen's dz)
- 95% confidence intervals where statistically meaningful

## Required visualizations implemented

- human vs model sensitivity comparison table
- human–model response correlation scatter plot
- disagreement-rate chart by perturbation
- disagreement-rate chart by image region
- severity–response curve
- human–model disagreement heatmap (Region × Perturbation)
- summary table/list of important findings

## Reproducibility

All analysis is recomputed from the SQLite-backed raw human responses, model scores, counterfactual metadata and region metadata. No chart uses manually entered result values.

## Important protocol note

The supplied project specification mentions an `Original / Modified / About the same` pairwise preference protocol. Module 3 currently uses a blind Best-of-four protocol plus 1–5 ratings. Module 4 therefore reports a derived original-vs-counterfactual Best-vote rate; it does not invent an `About the same` rate.

## Disagreement threshold

A candidate is counted as a substantial human–model disagreement when the absolute human/model score gap is at least **0.75 on the 1–5 scale**. The threshold is returned in the API output for reproducibility.

## Tests

Backend test target:

```text
24 passed
```

## Git checkpoint

After local UI validation:

```powershell
git add .
git commit -m "feat(module-4): add PDF-aligned quantitative analysis and visualizations"
git tag v4.2-module-4-pragya-context
git push origin main
git push origin v4.2-module-4-pragya-context
```


## v4.2 additions
- VLM-assisted structured context extraction with fallback
- custom original image + prompt input
- exact Original / Modified / About-the-same pairwise protocol
- raw pairwise preference storage and severity-specific rates
- context propagation into Module 4 error reports
- disagreement aggregation by image category/style
- 29 backend tests passing
