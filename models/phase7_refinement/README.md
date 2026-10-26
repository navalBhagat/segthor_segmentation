# Local inference models

Model weights are stored separately and are not included in Git. Place the
3-slice ENet and its trained trachea, esophagus and heart refiners in local
folders named `seed-42`, `seed-123` and `seed-2026` here. Each folder should
contain `enet.pt`, `trachea.pt`, `esophagus.pt` and `heart.pt`. These files
are ignored by Git.

Use all four checkpoints from the same seed together. Aorta keeps the ENet
prediction. The ENet weights are the supplied checkpoints; we trained only
the refiners.

We trained all nine refiners for 25 epochs and saved the epoch with the
highest mean patient-level 3D organ Dice in the original scan space, before
postprocessing. We used patients 01, 13, 22, 28 and 30 for validation and
checkpoint selection, so the reported scores are validation results.

| Seed | Trachea epoch | Esophagus epoch | Heart epoch |
| --- | ---: | ---: | ---: |
| 42 | 23 | 23 | 18 |
| 123 | 9 | 24 | 18 |
| 2026 | 20 | 21 | 12 |

See the [main README](../../readme.md#inference-with-per-organ-checkpoints) for
preprocessing, inference and postprocessing commands. From the repository root:

```bash
MODEL=models/phase7_refinement/seed-42
python inference.py --weights "$MODEL/enet.pt" \
  --trachea-checkpoint "$MODEL/trachea.pt" \
  --esophagus-checkpoint "$MODEL/esophagus.pt" \
  --heart-checkpoint "$MODEL/heart.pt" \
  --data-root data/SEGTHOR_TEST --net enet25d --slices 3 --slice-step 1 \
  --dest results/refined-seed42 --device cuda
```

Change `seed-42` to `seed-123` or `seed-2026` to use another seed. We evaluated
the three seeds separately, without an ensemble.
