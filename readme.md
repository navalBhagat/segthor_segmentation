# SegTHOR organ segmentation — Group 15

Three-slice ENet predicts the central CT slice. Separate trachea, esophagus and heart refiners correct its predictions using original-CT crops, preserving other organ labels. Aorta retains ENet. Outputs are reconstructed in the original NIfTI geometry, then postprocessed.

## Setup and data

Run commands from the repository root with Python 3.11 and the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Extract the course datasets into this layout:

```text
data/segthor_train/train/Patient_XX/Patient_XX.nii.gz
data/segthor_train/train/Patient_XX/GT.nii.gz
data/segthor_test/test/Patient_XX.nii.gz
```

Weights are stored locally: place `enet.pt`, `trachea.pt`, `esophagus.pt` and `heart.pt` from the **same seed** in `models/phase7_refinement/seed-123/` (or seed 42/2026). See [checkpoint setup](models/phase7_refinement/README.md). Inference needs original CTs, but no ground-truth masks.

## ENet training

Prepare full-field, 1 mm isotropic, 256 × 256 slices, then train with strong augmentation, CE + Dice and Adam (learning rate 0.0005):

```bash
python preprocess.py --source_dir data/segthor_train \
  --dest_dir data/SEGTHOR_new --shape 256 256 \
  --hu_min -1000 --hu_max 500 --mm_per_px 0 --target_spacing 1 1 1 \
  --retains 5 --seed 0 --process 1
python main.py --dataset SEGTHOR --data-root data/SEGTHOR_new \
  --mode full --net enet25d --slices 3 --slice-step 1 \
  --loss ce_dice --augmentation strong --optimizer adam \
  --seed 42 --num_workers 8 --epochs 25 --gpu \
  --val_3d --val_3d_every 1 --dest results/enet-s42
```

Repeat with seeds 123 and 2026 and separate destinations. Validation patients are **01, 13, 22, 28 and 30**; retain `splits.json` and `preprocess.json`. ENet selects `bestweights.pt` by native-volume foreground Dice; `summary.json` records its score and epoch.

## Organ refinement

The ENet stays frozen. Each organ refiner trains for 25 epochs and selects `best.pt` by patient-equal native-volume organ Dice. Trachea/esophagus use batch Dice in the training loss; heart uses per-image Dice. To reproduce training:

```bash
python experiments/trachea_data.py --archive /path/to/segthor-official-train.zip \
  --dest results/official40-cache --workers 4 --max-seconds 3600
python experiments/phase7_refinement.py --stage all \
  --data-cache results/official40-cache \
  --checkpoint models/phase7_refinement/seed-123/enet.pt --seed 123 \
  --dest results/phase7-study/seed-123 --epochs 25 \
  --checkpoint-selection best --workers 3 --device cuda --max-seconds 14000
```

Repeat with the matching ENet for each seed. Existing trained checkpoints can be used directly.

## Postprocessing

Apply the same fixed cleanup to validation and test: remove esophagus/trachea fragments below 1 mL while preserving the largest component, retain only the largest aorta component, and leave heart unchanged. No closing or hole filling is used. **Inference does not apply cleanup automatically.**

```bash
python postprocess.py --src results/refined-test-s123/volumes \
  --dest results/refined-test-s123/volumes_latest \
  --min_ml 1 --min-classes 1 3 --largest --largest-classes 4
```

For validation, replace the source and destination with the validation volume folders. The submission job applies this command to both sets automatically.

<a id="inference-with-per-organ-checkpoints"></a>

## Test inference

Preprocess test scans, run the matching four checkpoints, then clean the predictions:

```bash
MODEL=models/phase7_refinement/seed-123
OUT=results/refined-test-s123
python preprocess.py --source_dir data/segthor_test \
  --dest_dir data/SEGTHOR_TEST --test_only --shape 256 256 \
  --hu_min -1000 --hu_max 500 --mm_per_px 0 --target_spacing 1 1 1
python inference.py --weights "$MODEL/enet.pt" \
  --trachea-checkpoint "$MODEL/trachea.pt" \
  --esophagus-checkpoint "$MODEL/esophagus.pt" \
  --heart-checkpoint "$MODEL/heart.pt" \
  --data-root data/SEGTHOR_TEST --net enet25d --slices 3 --slice-step 1 \
  --dest "$OUT" --device cuda
python postprocess.py --src "$OUT/volumes" --dest "$OUT/volumes_latest" \
  --min_ml 1 --min-classes 1 3 --largest --largest-classes 4
```

Use a fresh output directory. Cleaned outputs are in `volumes_latest/`, with labels 0–4. Omit refiner flags for ENet alone. Use `--device cpu` or `mps` where appropriate. For validation + test inference and archiving on Snellius, use the job below.

## Submission

For standalone pickle export and inference:

```bash
python segthor_pipeline.py --model-dir models/phase7_refinement/seed-123 \
  --output results/bestmodel-group-15.pkl
python inference.py --pipeline results/bestmodel-group-15.pkl \
  --data-root data/SEGTHOR_TEST --dest results/pickle-test --device cuda
```

Apply the same postprocessing command afterward. The pickle needs the submitted Python code, but no separate organ checkpoint files.

<a id="completed-3-slice-enet-refinement-results"></a>

## Results

Native-volume metrics, mean ± sample SD across seeds **42, 123 and 2026**, weighting patients and foreground organs equally:

| Pipeline | Dice ↑ | HD95 mm ↓ | Absolute RVD ↓ |
| --- | ---: | ---: | ---: |
| ENet | 0.8213 ± 0.0006 | 10.83 ± 1.58 | 0.1045 ± 0.0283 |
| ENet + refinement | 0.8617 ± 0.0034 | 8.93 ± 1.59 | 0.0897 ± 0.0081 |
| ENet + cleanup | 0.8233 ± 0.0016 | 9.05 ± 0.22 | 0.0993 ± 0.0291 |
| **ENet + refinement + cleanup** | **0.8649 ± 0.0043** | **6.78 ± 0.21** | **0.0861 ± 0.0098** |

Raw organ Dice improves **0.6473 → 0.7467** for esophagus, **0.8383 → 0.8948** for trachea and **0.9186 → 0.9240** for heart; aorta remains **0.8811**. The historical anisotropic baseline scored **0.7158 Dice**. These are validation results: the same five patients were reused for checkpoint selection and evaluation, so they do not estimate untouched test performance.

See the [results notebook](notebooks/result_aggrestion.ipynb), [metric tables](evaluation/phase7_refinement/README.md), [graphs](docs/figures/phase7_refinement/README.md) and [branch summary](docs/research/BRANCH_SUMMARY.md) for details. Regenerate graphs with:

```bash
python experiments/phase7_refinement_report.py \
  --snapshot notebooks/data/phase7_refinement_results.json --dest results/phase7-report
```

Original course and submission instructions: [old_readme.md](old_readme.md).
