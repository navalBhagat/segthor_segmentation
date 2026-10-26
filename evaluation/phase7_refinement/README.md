# 3-slice ENet refinement evaluation

This folder contains the 3D scores used in our [graphs](../../docs/figures/phase7_refinement/README.md). Model weights and prediction volumes are kept separately from Git. See the [checkpoint setup](../../models/phase7_refinement/README.md) for using local weights.

## Files

| File or directory | Contents |
| --- | --- |
| [patient_metrics.csv](patient_metrics.csv) | 240 rows: four organs × five patients × three seeds × four prediction stages |
| [summary.csv](summary.csv) | Per-organ and four-organ means, sample SD, and individual seed means for Dice, HD95 and absolute RVD |
| [historical_baseline_summary.csv](historical_baseline_summary.csv) | Previously reported original baseline means and SD |

The [results JSON](../../notebooks/data/phase7_refinement_results.json) also contains the training histories and selected epochs.

## Prediction stages

| Stage | Predictions |
| --- | --- |
| `raw_enet` | 3-slice ENet, before cleanup |
| `raw_refined` | 3-slice ENet + refinement, before cleanup |
| `latest_enet` | 3-slice ENet with the fixed cleanup policy |
| `latest_refined` | 3-slice ENet + refinement with the same cleanup policy |

Each stage contains patients 01, 13, 22, 28 and 30 for all three seeds. For `latest_*`, we remove esophagus and trachea components below 1 mL but always keep the largest component. We leave the heart unchanged and keep only the largest aorta component. The Dice comparison graphs use the raw predictions.

The local NIfTI predictions use labels **0 background, 1 esophagus, 2 heart, 3 trachea, 4 aorta**. CT scans and ground-truth masks are also kept outside Git. Use the original SEGTHOR CT scans for overlays and the ground-truth masks to recalculate the scores.

## Metric definitions

We calculate scores on the complete 3D volumes in the original patient geometry. Dice measures overlap, HD95 is in millimetres, and signed RVD is `(predicted volume - ground-truth volume) / ground-truth volume`. RVD is a ratio, not a percentage. For absolute RVD, we take the absolute value for each patient and organ before averaging.

For the summary graphs, we first average the five patients within each seed, then report the mean and sample SD of the three seed means. Overall scores give each organ equal weight. These are validation results: we used the same five patients to select the checkpoints.

The patient-level plots show 15 observations per stage: the same five patients for three seeds. These are repeated measurements, not 15 different patients.

For the original baseline, we only have the previously reported, rounded summaries here. Its patient scores and prediction volumes are not included. The gain on slide 17 is relative to the original baseline; the separate refinement comparison shows the gain over 3-slice ENet.

## Use the files

Run this command from the repository root to regenerate the graphs from the saved results:

```bash
python experiments/phase7_refinement_report.py \
  --snapshot notebooks/data/phase7_refinement_results.json \
  --dest results/phase7-report
```

This includes the patient-level Dice, HD95 and signed RVD plots. No weights or prediction volumes are needed to regenerate them.

To recalculate the 3D scores from local predictions for one seed and stage:

```bash
python metrics_3d.py \
  --volumes_root /path/to/local/predictions/seed-42/raw_refined \
  --gt_root /path/to/segthor/train \
  --dest results/recomputed/seed-42/raw_refined \
  --val_patients 1,13,22,28,30 --K 5
```

Ground truth can be stored as `Patient_01/GT.nii.gz` directories or flat `Patient_01.nii.gz` files. Change the seed and stage paths for the other predictions. Keep the resulting NPZ files in the local `results/` folder. The [main README](../../readme.md#regenerate-results-and-presentation-figures) has more examples.
