# Module 2 checkpoint — Sanyojita Priyadarshani Singh

**Checkpoint:** `v2.0-module-2-sanyojita`

Module 2 begins only after the frozen Module 1 v1.4 handoff exists.

## Responsibility

Module 2 performs controlled counterfactual generation:

1. Receive the Module 1 `image_id` and selected `region_id`.
2. Reuse Module 1 segmentation mask / polygon / bounding box.
3. Apply exactly one perturbation type at a time.
4. Generate `subtle`, `moderate`, and `strong` variants.
5. Save counterfactuals as lossless PNG.
6. Validate that pixels outside the selected region have maximum absolute difference `0`.
7. Store perturbation parameters and validation metadata.
8. Export a stable Module 3 handoff containing original image, region, perturbation metadata and all counterfactuals.

## Supported perturbations

- brightness reduction
- contrast reduction
- saturation reduction
- Gaussian blur
- Gaussian noise
- over-sharpening
- colour / hue shift
- pixelation
- structural distortion

## Boundary

Module 2 does **not** perform aesthetic-model scoring, human voting, or human–model agreement analysis. Those belong to Modules 3 and 4.
