# PDF requirement coverage after v4.3 context-quality + protocol enhancement

## Strongly covered

- AI-generated image + prompt input, including AGIQA-3K and custom upload.
- Reproducible original/modified image mapping and metadata.
- VLM-assisted structured context extraction with safe fallback.
- Objects, attributes, relationships and scene/context JSON.
- YOLO region detection/segmentation and full-image fallback.
- Controlled perturbation engine with nine perturbation types.
- Subtle / moderate / strong counterfactuals and exact outside-region preservation checks.
- Model aesthetic scoring and score deltas.
- Blind 1–5 human ratings + Best vote.
- Exact `Original / Modified / About the same` pairwise human protocol for every severity.
- Raw human-response storage.
- Human–model comparison, directional agreement, Pearson, Spearman, MAE, RMSE.
- Disagreement rate, severity-response slopes, effect size and 95% confidence intervals.
- Sensitivity analysis by perturbation, severity and region.
- Project-level image-category/style aggregation.
- Correlation plot, disagreement charts, severity curves, heatmap and findings summary.
- Context-aware disagreement/error reporting.
- JSON/CSV/database outputs, requirements files, automated tests and reproducible run instructions.

## Remaining limitations

- The lightweight BLIP model contributes visual caption context; complex open-ended relational reasoning may still rely on prompt structure rather than a very large MLLM.
- The project implements a representative controlled-perturbation set rather than every example perturbation named in the specification.
- Statistical conclusions still require an adequate number of real participants and experimental batches; software support alone is not a substitute for collected study data.

## Estimated implementation coverage

After local runtime validation, this build is designed to satisfy approximately **94–97%** of the supplied functional and quantitative specification. This is an engineering estimate rather than a formal academic grading score.


## Context-quality refinement

The implementation now records source-separated detector objects, prompt entities, VLM entities, structured attributes, relationships, scene tags, and detector reliability. This improves the contextual-error-report requirement by preventing descriptive words from being misreported as confirmed objects.
