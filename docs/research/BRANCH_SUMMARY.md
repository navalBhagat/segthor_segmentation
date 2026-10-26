# 3-slice ENet + organ refinement — 11 October 2026

The current result uses the supplied **3-slice ENet checkpoints**, frozen independently for seeds **42, 123 and 2026**. All nine trachea, esophagus and heart refiners completed **25 epochs** on the original 35 training patients. Predictions run **trachea → esophagus → heart**, preserving other organ labels; aorta retains ENet.

| Pipeline | Native 3D Dice ↑ | HD95 (mm) ↓ | Absolute RVD ↓ |
| --- | ---: | ---: | ---: |
| 3-slice ENet, raw | 0.8213 ± 0.0006 | 10.83 ± 1.58 | 0.1045 ± 0.0283 |
| 3-slice ENet + refinement, raw | **0.8617 ± 0.0034** | **8.93 ± 1.59** | **0.0897 ± 0.0081** |
| 3-slice ENet + existing cleanup | 0.8233 ± 0.0016 | 9.05 ± 0.22 | 0.0993 ± 0.0291 |
| 3-slice ENet + refinement + existing cleanup | **0.8649 ± 0.0043** | **6.78 ± 0.21** | **0.0861 ± 0.0098** |

Raw Dice improves by **4.04 percentage points**; with the existing cleanup, it improves by **4.15 points**. Dice and HD95 improve for every seed. Absolute RVD improves on average but worsens for seed 42. The raw per-organ Dice changes are **0.6473 → 0.7467 esophagus**, **0.9186 → 0.9240 heart**, **0.8383 → 0.8948 trachea**, and **0.8811 → 0.8811 aorta**.

## Checkpoint selection

The new refiners validate after every epoch using **one Dice ratio over each complete native patient volume**, including errors outside the crop, then average the five patient scores equally. The best checkpoint is retained as `best.pt`; `last.pt` retains epoch 25. Crop placement and prediction inputs use ENet outputs and CT, without validation ground truth. Ground truth enters only scoring and training targets.

| Seed | Selected trachea epoch | Selected esophagus epoch | Selected heart epoch |
| --- | ---: | ---: | ---: |
| 42 | 23 | 23 | 18 |
| 123 | 9 | 24 | 18 |
| 2026 | 20 | 21 | 12 |

Epoch numbers above are one-based. Training-loss Dice reduction (batch for esophagus/trachea, per-image for heart) is separate from checkpoint selection.

With native 3D validation enabled in `main.py`, only epochs actually evaluated in 3D are eligible for selection, including the final epoch. Slice Dice selection requires explicit `--no_val_3d`.

## Evaluation and results

All current metrics use `metrics_3d.py` in the original patient geometry, with validation patients **01, 13, 22, 28 and 30**. Means weight patients and foreground organs equally; ± is sample standard deviation across three seed means. Repeated use of these patients for selection means these are validation results, not an untouched test estimate.

The supplied ENets re-evaluate at raw Dice **0.821322**, slightly different from the historically reported rounded **0.8212**. New gains use the exact matched re-evaluation. The fixed cleanup policy is applied equally to both pipelines: esophagus/trachea remove fragments below 1 mL while always preserving the largest component, heart is unchanged, and aorta retains its largest component. No new postprocessing search was performed.

- [Local checkpoint setup](../../models/phase7_refinement/README.md) and [per-organ inference instructions](../../readme.md#inference-with-per-organ-checkpoints).
- [Executed results notebook](../../notebooks/result_aggrestion.ipynb) and [portable patient-level snapshot](../../notebooks/data/phase7_refinement_results.json).
- [Regenerated presentation graphs](../figures/phase7_refinement/README.md), including the six affected slides, learning curves, selected epochs and four patient-level metric diagnostics.
- [Patient comparisons](final_pipeline_comparison.pdf) and [crop coverage audit](crop_failure_examples.pdf).

The notebook preserves the original upstream phase results and adds the executed matched refinement comparison. The committed snapshot contains patient metrics, all nine training histories, checkpoint selection and provenance; the report renderer regenerates the figures directly from it. Model weights, prediction volumes, full training archives and presentation copies remain outside Git.

The public inference entrypoint was checked on Patient 13 for every seed, both ENet-only and with all three refiners. All six outputs match the evaluated native predictions exactly, with zero differing voxels and unchanged native geometry.
