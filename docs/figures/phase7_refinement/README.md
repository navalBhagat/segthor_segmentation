# 3-slice ENet refinement figures

These figures compare 3-slice ENet with 3-slice ENet + refinement. We kept ENet frozen and trained trachea, esophagus and heart refiners for 25 epochs for each seed. We selected the best epoch using mean patient-level 3D Dice in the original scan space. Aorta uses the ENet prediction.

The data is in [phase7_refinement_results.json](../../../notebooks/data/phase7_refinement_results.json). For the summary figures, we first average over patients within each seed, then over seeds 42, 123 and 2026. Error bars show the sample standard deviation of the three seed means. The five validation patients were also used to select the checkpoints.

The [evaluation folder](../../../evaluation/phase7_refinement/README.md) contains the patient scores and summary tables in CSV format. Model weights and prediction volumes are stored separately.

| Presentation slide | Replacement figure | Vector copy |
| --- | --- | --- |
| 15: postprocessing | [Fixed cleanup effects](slide15_fixed_cleanup.png) | [PDF](slide15_fixed_cleanup.pdf) |
| 16: pipeline | [ENet and organ refinement](slide16_pipeline.png) | [PDF](slide16_pipeline.pdf) |
| 17: final Dice | [Dice comparison](slide17_dice.png) | [PDF](slide17_dice.pdf) |
| 18: takeaways | [Summary cards](slide18_summary.png) | [PDF](slide18_summary.pdf) |
| 19: other metrics | [HD95 and absolute RVD](slide19_other_metrics.png) | [PDF](slide19_other_metrics.pdf) |
| 23: organ refiner | [Matched refinement comparison](refinement_comparison.png) | [PDF](refinement_comparison.pdf) |

Other figures show [training loss](refiner_training_curves.png), [3D validation Dice](refiner_validation_curves.png), and [selected epochs](refiner_selected_epochs.png). PDF copies use the same filenames. File hashes are listed in [figure_manifest.json](figure_manifest.json).

On slide 17, Dice gain is the refined model's score minus the original baseline score. In the two-model refinement comparison, it is the refined score minus the 3-slice ENet score. The original baseline values are the previously reported rounded means.

The patient-level plots below use `plot.py` to show Dice versus HD95 and signed RVD. The summary plots above use absolute RVD.

| Predictions | Diagnostic figure | Vector copy |
| --- | --- | --- |
| 3-slice ENet, raw | [Patient metrics](patient_3d_metrics_raw_enet.png) | [PDF](patient_3d_metrics_raw_enet.pdf) |
| 3-slice ENet + refinement, raw | [Patient metrics](patient_3d_metrics_raw_refined.png) | [PDF](patient_3d_metrics_raw_refined.pdf) |
| 3-slice ENet + fixed cleanup | [Patient metrics](patient_3d_metrics_latest_enet.png) | [PDF](patient_3d_metrics_latest_enet.pdf) |
| 3-slice ENet + refinement + fixed cleanup | [Patient metrics](patient_3d_metrics_latest_refined.png) | [PDF](patient_3d_metrics_latest_refined.pdf) |

Each organ has 15 points: three seeds evaluated on the same five patients.

Both models use the same postprocessing: remove esophagus/trachea components below 1 mL while keeping the largest component, leave heart unchanged, and keep only the largest aorta component. We used the existing policy without repeating the policy search. Raw results are measured before postprocessing.

To use these figures in the presentation, insert the PNGs or pass a downloaded PowerPoint file to the report script with `--deck`.

The [main README](../../../readme.md#regenerate-results-and-presentation-figures) has the commands to regenerate the figures. The [results notebook](../../../notebooks/result_aggrestion.ipynb) contains the tables and plots.

The figures use the included [Inter font](fonts/Inter.ttf), distributed under the [SIL Open Font License](fonts/Inter-OFL.txt).
