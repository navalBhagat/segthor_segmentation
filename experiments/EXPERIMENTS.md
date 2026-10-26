> **Completed 3-slice ENet refinement:** the supplied frozen ENets for seeds 42, 123 and 2026 each receive trachea, esophagus and heart refiners trained for 25 epochs. Each organ uses its best mean patient-level native 3D Dice checkpoint. Matched raw Dice improves **0.8213 → 0.8617**; with the existing cleanup it improves **0.8233 → 0.8649**. See the [current training and inference commands](../readme.md#train-refiners-on-the-supplied-3-slice-enet-checkpoints), [results notebook](../notebooks/result_aggrestion.ipynb) and [final report](../docs/research/BRANCH_SUMMARY.md). The original experiment report below retains its own protocols and scores.

# Experiments

The starting point was the ENet provided with the course. The experiments
tested whether augmentation and neighboring CT slices could improve the
segmentation. A wider 3D U-Net with strong augmentation gave the best result.

## What changed

- **Augmentation:** during training, the code randomly rotates, moves and
  rescales the images, and changes their intensity. The mild setting uses
  geometry, brightness changes (gamma) and noise. The strong setting uses
  larger changes and also adds contrast changes and blur. Images and masks
  get the same geometric changes, so they stay aligned. For a stack, the same
  rotation, shift and scale are applied to every slice, within the image plane.
  Intensity changes only affect the images. Validation images are left as they
  are.
- **2.5D ENet:** ENet can take one, three or five slices as input. With three
  or five, it still predicts only the middle slice. The others give it some
  context. This mainly uses the loader in `dataset.py` and a small change to
  ENet's first convolution. The one-slice architecture stays the same.
- **3D U-Net:** `UNet3D.py` and `volume_dataset.py` add a model that works on
  blocks of slices and predicts a mask for every slice in the block. The
  experiments tried different block depths and model widths. The loader keeps
  slices in order and makes sure a block does not mix patients.
- **Comparing runs:** `main.py` and `measurement.py` record the settings,
  training time, and Dice scores per patient and organ. Fixed random seeds
  make it possible to repeat runs.

Augmentation runs **online**: random transformations are sampled whenever a
training example is loaded. The settings in `dataset.py` are:

| Transformation | Mild | Strong |
| --- | --- | --- |
| Rotation | Up to ±10° | Up to ±15° |
| Horizontal/vertical shift | Up to ±5% of image width/height | Up to ±10% |
| Zoom (scale factor) | 0.9–1.1 | 0.8–1.2 |
| Brightness adjustment (gamma) | 0.9–1.1 | 0.7–1.5 |
| Gaussian noise (standard deviation) | 0–0.02 | 0–0.04 |
| Contrast (factor) | Not used | 0.8–1.2 |
| Gaussian blur | Not used | Sigma 0.5–1.0 pixels, 7 × 7 kernel |

Rotation, shift and zoom form one geometric operation, applied with 50%
probability for mild and 70% for strong. Gamma and noise are applied with
30% and 20% probability, respectively. Strong also applies contrast with
30% probability and blur with 20%. These component choices are independent;
multiple changes can occur together. Noise levels use image intensities
normalized to 0–1.

Training still uses the original cross-entropy loss and Adam optimizer, with a
constant learning rate of `0.0005`. The small change in `losses.py` lets the
same loss handle 3D outputs. These experiments do not use Dice loss or a
learning-rate schedule.

## What worked

These are the main comparisons, averaged over seeds 0, 1 and 2. Each used the
same 15 training patients and five validation patients, with 60 validation
rounds and 21,180 training updates.

| Model | Selected Dice ± SD | Final Dice ± SD | Selected gain over plain ENet, points | Train min | Peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| One-slice ENet, no augmentation | 0.7357 ± 0.0399 | 0.6996 ± 0.0253 | — | 74.3 | 0.769 |
| One-slice ENet, strong augmentation | 0.8636 ± 0.0079 | 0.8505 ± 0.0037 | +12.78 | 75.0 | 0.769 |
| Three-slice ENet, strong augmentation | 0.8523 ± 0.0069 | 0.8445 ± 0.0043 | +11.66 | 75.3 | 0.772 |
| Wide 3D U-Net, no augmentation | 0.8602 ± 0.0085 | 0.8617 ± 0.0153 | +12.45 | 202.0 | 2.430 |
| **Wide 3D U-Net, strong augmentation** | **0.8875 ± 0.0069** | **0.8750 ± 0.0244** | **+15.18** | **206.7** | **2.430** |

Dice measures how much the prediction overlaps the mask; 1 means a perfect
match. The table uses Dice for whole patient volumes, averaged equally over
foreground classes and patients. "Selected" means the saved checkpoint with
the best average foreground **slice** Dice during validation. "Final" means
the last checkpoint. Because those averages are calculated differently, the
final volume score can be higher than the selected one.

`± SD` shows the sample standard deviation across the three seeds. A gain of
2.4 points means a Dice increase of 0.024. Training minutes do not include
validation or time waiting for a GPU. Memory is peak GPU memory used by
PyTorch. Mild one-slice ENet, which is in the comparison table below, scored
0.802 selected and 0.798 final.

Strong augmentation made a big difference for ENet. Adding neighboring
slices did not beat the strong one-slice model on average. The wide 3D model
with strong augmentation did better overall, mainly on the esophagus and
trachea, but it took about 2.8 times as long to train as strong ENet. Heart
performance was slightly worse at selection, and one 3D run had a large heart
drop at the end.

Here is the organ comparison between **strong one-slice ENet → strong wide
3D**. Each score is averaged over the same five patients and three seeds.

| Organ | Selected Dice | Change, points | Final Dice | Change, points |
| --- | ---: | ---: | ---: | ---: |
| Esophagus | 0.7435 → 0.7801 | +3.66 | 0.7232 → 0.7622 | +3.90 |
| Heart | 0.9166 → 0.9119 | −0.47 | 0.9111 → 0.8618 | −4.93 |
| Trachea | 0.7942 → 0.8581 | +6.39 | 0.7678 → 0.8759 | +10.81 |

Other experiments tried different slice counts, 3D sizes, parts of the
augmentation policy, and training lengths. The table shows what each one
answered. Unless marked "final", these are **selected** patient-volume Dice
scores.
Rows marked "seed 0" are one exploratory run, while rows marked "three seeds"
are averages of three runs.

| What was tried | Result | What it showed |
| --- | --- | --- |
| Augmenting one-slice ENet | **Three seeds, 60 rounds:** no augmentation 0.736, mild 0.802, strong 0.864. | Strong helped in all three seeds. |
| Giving ENet three slices | **Three seeds, 60 rounds:** 0.733 without augmentation, 0.819 with mild, 0.852 with strong. Strong one-slice ENet got 0.864. | Three slices helped the mild setting, but did not beat strong one-slice ENet. |
| Using more or more widely spaced slices | **Seed 0:** after 40 rounds, five adjacent slices scored 0.757 and five spaced slices scored 0.720. At 60 rounds, strong three-slice step 2 was 0.14 points below strong one-slice ENet. | More slices were not automatically better. Several 60-round five-slice runs did not finish. |
| Making 3D deeper and wider | **Seed 0, 60 rounds:** narrow depth 16 scored 0.779, narrow depth 32 scored 0.826, and wide depth 32 scored 0.856. The later three-seed plain-wide average was 0.860. | The wider model looked worth testing further. A deeper block also shows the model more slices per update. |
| Training for longer | **Seed 0, plain wide 3D:** selected Dice was 0.857 at 40 rounds and 0.856 at 60 rounds. | More rounds did not always help. |
| Adding strong augmentation to wide 3D | **Three seeds, 60 rounds:** 0.888, compared with 0.860 for plain wide 3D and 0.864 for strong ENet. Final 3D beat ENet in two of three seeds. | This gave the best selected result, with extra training cost and the heart limitation above. |

There are two things to keep in mind when reading the scores. The prepared
masks contain no aorta labels, although the network still has five output
classes: background and four organs. An empty aorta prediction gets Dice 1
and raises the overall average. The same five validation patients were used
throughout, so these results do not come from a separate test set.

## Before running

The validation patients are 01, 11, 15, 17 and 19. Using a different split or
the full official dataset would be a different experiment.

The commands below expect the prepared data here:

```text
data/SEGTHOR/
  train/
    img/
    gt/
  val/
    img/
    gt/
```

`img` contains CT slices and `gt` contains their masks, with matching names
such as `Patient_01_0000.png`. Keep the patient names and slice numbers. If
the data is stored elsewhere, replace `data/SEGTHOR` after `--data-root` in
the commands.

## Running the final model

This trains the wide 3D U-Net with strong augmentation for seed 0:

```bash
python main.py --dataset SEGTHOR --mode full \
  --model unet3d --depth 32 --widths 16 32 64 --pooling isotropic \
  --augmentation strong --batch-size 1 --train-steps 353 --epochs 60 \
  --seed 0 --workers 5 --data-root data/SEGTHOR \
  --dest results/wide3d-strong-s0 --gpu
```

`--depth 32` means each input block contains 32 slices. `--widths 16 32 64`
sets the number of feature channels at the three levels of the U-Net. These
are the wider settings used for the final result.

For 3D, `--train-steps 353` means 353 training updates before each validation.
With `--epochs 60`, this gives 21,180 updates. Here an epoch is a validation
interval, not a full pass through all possible 3D blocks. It matches the
number of updates in the ENet runs, although 3D sees more slices per update.

To repeat the comparison, also run with `--seed 1` and `--seed 2`, changing
the destination to `results/wide3d-strong-s1` and `results/wide3d-strong-s2`.
The seed changes the initial weights, data order and random augmentation;
the patients stay the same. These are three separately trained models.

Use a new destination for each run. Training starts from scratch; pointing
to an old result folder does not resume it and can overwrite its outputs.

## Running the other comparisons

For one-slice ENet with strong augmentation:

```bash
python main.py --dataset SEGTHOR --mode full \
  --model enet --input-slices 1 --slice-step 1 \
  --augmentation strong --batch-size 8 --epochs 60 \
  --seed 0 --workers 5 --data-root data/SEGTHOR \
  --dest results/enet-strong-s0 --gpu
```

From that command:

- Use `--augmentation none` for the baseline, or `mild` for mild augmentation.
- Use `--input-slices 3` or `5` for 2.5D. A three-slice input uses the slice
  before, the middle slice and the slice after. The target is the middle
  slice's mask. Train a separate model for each input count.
- Use `--slice-step 2` to take every other slice instead of consecutive ones.
  Missing neighbors at a patient's boundaries are filled by repeating the
  nearest available slice.
- Change the destination name to match the settings, and repeat with seeds
  1 and 2 for a comparison across three runs.

For the other 3D settings, start from the final-model command and replace
the options below. Keep batch size 1 and 353 training updates per validation.

| Experiment | Options to use |
| --- | --- |
| Wide 3D without augmentation | `--augmentation none` |
| Smaller 3D model, 16 slices | `--depth 16 --widths 8 16 32` |
| Smaller 3D model, 32 slices | `--depth 32 --widths 8 16 32` |

Choose `none`, `mild` or `strong` for the augmentation being compared.
`--input-slices` and `--slice-step` belong to ENet; 3D uses `--depth`.

For an augmentation component comparison, adding
`--augmentation-components affine` keeps only the geometric changes.
Other available components are `gamma`, `noise`, `contrast` and `blur`;
contrast and blur belong to the strong setting. Leave this option out to
use the full policy, as in the final model.

For a quick check before a long run, change `--epochs 60` to `--epochs 1`
and add `--debug --max-updates 1`, with a separate destination. For 3D, also
change `--train-steps 353` to `--train-steps 1`. This checks one training update
and validation on a small subset. Its score should not be compared with the
full experiments.

## Where to find the results

Each destination folder contains:

- `run.json`: the settings, seed, data split and software versions.
- `epochs.json`: the scores and timings after each validation.
- `summary.json`: the selected and final scores, total updates and timings.
- `bestweights.pt` and `finalweights.pt`: the selected and final model weights.
- `best_epoch/val/`: predicted mask PNGs from the selected checkpoint.
- `iter000/val/`, `iter001/val/`, etc.: predictions from each validation round.

In `summary.json`, the scores used in the table are
`selected.patient_volume_dice.foreground_mean` and
`final.patient_volume_dice.foreground_mean`. Check `epochs_completed` is 60
and `updates` is 21,180 before treating a run as a completed comparison.
Average the three seed scores; do not just pick the highest one. It is also
useful to check `per_organ`, because a good overall score can hide a worse
result for one organ.
