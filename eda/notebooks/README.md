# Data Exploration Summary

Notebook: `notebooks/data_exploration.ipynb`

## Dataset

- Data are 3D thoracic CT volumes in `.nii.gz` format with matching 3D GT masks.
- Current split used in the notebook:
  - Train: 15 patients
  - Validation: 5 patients
- CT and GT shapes match for every checked patient.
- In-plane size is always `512 x 512`; number of axial slices varies by patient.

## Train Split Findings

- Patients loaded: 15
- Total GT slices: 2817
- Average slices per patient: 187.80
- Average image shape: `(512, 512, 187)`
- Average voxel spacing: approximately `(1.00, 1.00, 2.33)` mm
- Orientation: all checked patients use `('L', 'P', 'S')`
- Image dtype differs across patients: some `int16`, some `int32`
- GT dtype: `uint8`
- Average CT intensity range: `-1000.0` to `7526.87`
- GT values present: `{0, 1, 2, 3}`
- Label `4` is missing from all checked training GTs.

## Validation Split Findings

- Patients loaded: 5
- Total GT slices: 915
- Average slices per patient: 183.00
- Average image shape: `(512, 512, 183)`
- Average voxel spacing: approximately `(1.02, 1.02, 2.40)` mm
- Orientation: all checked patients use `('L', 'P', 'S')`
- Image dtype differs across patients: some `int16`, some `int32`
- GT dtype: `uint8`
- Average CT intensity range: `-1000.0` to `15204.80`
- GT values present: `{0, 1, 2, 3}`
- Label `4` is missing from all checked validation GTs.

## Slice-Level Findings

- Background-only slices:
  - Train: 36.42%
  - Validation: 36.39%
- Slices containing multiple foreground classes:
  - Train: 50.02%
  - Validation: 50.82%
- Percentage of slices containing each foreground class:
  - Class 1: train 63.51%, validation 63.61%
  - Class 2: train 22.51%, validation 22.30%
  - Class 3: train 27.76%, validation 28.63%
  - Class 4: train 0.00%, validation 0.00%

## Class Imbalance

- Average foreground voxel ratio per patient:
  - Class 1: train 0.00369, validation 0.00410
  - Class 2: train 0.03484, validation 0.03189
  - Class 3: train 0.00123, validation 0.00165
  - Class 4: absent
- Class 2 is much larger than classes 1 and 3.
- Class 3 is the smallest foreground class by voxel ratio.

## Z-Axis Presence

- Mean number of slices containing each class:
  - Class 1: train 119.27, validation 116.40
  - Class 2: train 42.27, validation 40.80
  - Class 3: train 52.13, validation 52.40
  - Class 4: absent
- Class 1 appears over the widest z-range.
- Classes 2 and 3 appear in fewer slices, suggesting stronger slice-level imbalance.

## Important Interpretation

- The baseline code expects `K=5`, i.e. background plus four foreground classes.
- The current `segthor_part1` data contains only labels `0, 1, 2, 3`.
- This likely explains why `k=4` Dice can appear artificially perfect: class 4 is absent in GT and may also be absent in predictions.
- 3D per-patient metrics should report absent classes explicitly instead of silently averaging them.
