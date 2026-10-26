# Final postprocessing: chooses a cleanup policy per organ on the validation set,
# then applies it to the test predictions.
#
# Run from the repository root, after training and after test inference:
#     python postprocess_final.py
#
# Before running, set FINAL_CONFIG below to the configuration whose checkpoint was used
# for test inference, e.g. 'enet_strong_aug_ce_dice' or 'enet25d_3s_step1_strong_aug_ce_dice'.
#
# Expects:
#   results/segthor_new/final_<FINAL_CONFIG>_25ep_s<seed>/volumes/best_epoch/val/Patient_XX.nii.gz
#       Validation predictions written by training, for seeds 42, 123 and 2026.
#   data/segthor_train/train/Patient_XX/GT.nii.gz
#       Ground truth of the validation patients.
#   results/segthor_new/final_inference/volumes/Patient_XX.nii.gz
#       Raw test predictions, written by:
#       python inference.py --weights <checkpoint> --dest results/segthor_new/final_inference ...
#
# Writes to results/segthor_new/postprocessing_final/:
#   policy.json        the chosen cleanup per organ
#   held_out.csv       gain measured on patients the choice was not made on; report this one
#   summary.csv        every policy per organ, averaged over patients
#   per_patient.csv    every policy, seed, patient and organ
#   volumes/           the postprocessed test predictions, the final submission
#
# How the policy is chosen, per organ:
#   1. Try every combination of removal (none, largest piece only, or drop pieces below
#      0.1/0.25/0.5/1/2/5 mL), closing (0, 1, 2 voxels) and hole filling (off, on).
#   2. Average each patient over the seeds and compare against the raw prediction.
#   3. Reject policies where any patient loses more than 0.02 Dice.
#   4. Take the simplest policy whose mean HD95 change is within one standard error of the best.
#   5. Repeat with each patient held out to estimate the gain without tuning on that patient.

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from metrics_3d import dice_3d, hd95_3d, parse_patient_number, patient_volume
from postprocess import close_gaps, fill_holes, remove_pieces

FINAL_CONFIG = 'SET_ME'
SEEDS = [42, 123, 2026]
RESULTS = Path('results/segthor_new')
GT_ROOT = Path('data/segthor_train/train')
TEST_VOLUMES = RESULTS / 'final_inference' / 'volumes'
DEST = RESULTS / 'postprocessing_final'

REMOVALS = ['none', 'largest', 0.1, 0.25, 0.5, 1, 2, 5]
CLOSES = [0, 1, 2]
FILLS = [False, True]
MARGIN = 5
MAX_DICE_DROP = 0.02
LABELS = {'esophagus': 1, 'heart': 2, 'trachea': 3, 'aorta': 4}


def policy_name(rule):
    return f"removal={rule['removal']} close={rule['close']} fill={rule['fill']}"


def all_policies():
    policies = {}
    for removal in REMOVALS:
        for close in CLOSES:
            for fill in FILLS:
                rule = {'removal': removal, 'close': close, 'fill': fill}
                policies[policy_name(rule)] = rule
    return policies


POLICIES = all_policies()
RAW = policy_name({'removal': 'none', 'close': 0, 'fill': False})


def operation_count(rule):
    count = 0
    if rule['removal'] != 'none':
        count += 1
    if rule['close'] > 0:
        count += 1
    if rule['fill']:
        count += 1
    return count


def apply_policy(pred, policy, voxel_ml):
    cleaned = pred.copy()
    for organ, rule in policy.items():
        if rule['removal'] == 'largest':
            remove_pieces(cleaned, LABELS[organ], np.inf)
        elif rule['removal'] != 'none':
            remove_pieces(cleaned, LABELS[organ], rule['removal'] / voxel_ml)
    for organ, rule in policy.items():
        if rule['close'] > 0:
            close_gaps(cleaned, LABELS[organ], rule['close'])
        if rule['fill']:
            fill_holes(cleaned, LABELS[organ])
    return cleaned


def bounding_box(mask):
    box = []
    for coords, size in zip(np.nonzero(mask), mask.shape):
        start = max(coords.min() - MARGIN, 0)
        stop = min(coords.max() + MARGIN + 1, size)
        box.append(slice(start, stop))
    return tuple(box)


def evaluate(seed, patient, pred_path, gt_path):
    pred = np.asarray(nib.load(pred_path).dataobj, dtype=np.uint8)
    gt_image = nib.load(gt_path)
    gt = np.asarray(gt_image.dataobj, dtype=np.uint8)
    spacing = tuple(float(s) for s in gt_image.header.get_zooms()[:3])
    voxel_ml = np.prod(spacing) / 1000

    rows = []
    for organ, organ_label in LABELS.items():
        box = bounding_box((pred == organ_label) | (gt == organ_label))
        pred_box = pred[box]
        gt_mask = (gt[box] == organ_label).astype(np.uint8)
        for name, rule in POLICIES.items():
            cleaned = apply_policy(pred_box, {organ: rule}, voxel_ml)
            mask = (cleaned == organ_label).astype(np.uint8)
            rows.append({
                'seed': seed,
                'patient': patient,
                'organ': organ,
                'policy': name,
                'dice': dice_3d(mask, gt_mask, 2)[1],
                'hd95_mm': hd95_3d(mask, gt_mask, 2, spacing)[1],
            })
    return rows


def add_deltas(results):
    raw = results[results['policy'] == RAW]
    raw = raw[['seed', 'patient', 'organ', 'dice', 'hd95_mm']]
    results = results.merge(raw, on=['seed', 'patient', 'organ'], suffixes=('', '_raw'))
    results['delta_dice'] = results['dice'] - results['dice_raw']
    results['delta_hd95_mm'] = results['hd95_mm'] - results['hd95_mm_raw']
    return results


def summarise(results):
    columns = ['delta_dice', 'delta_hd95_mm']
    by_patient = results.groupby(['organ', 'policy', 'patient'], as_index=False)[columns].mean()
    summary = by_patient.groupby(['organ', 'policy'], as_index=False).agg(
        mean_delta_hd95_mm=('delta_hd95_mm', 'mean'),
        se_delta_hd95_mm=('delta_hd95_mm', 'sem'),
        mean_delta_dice=('delta_dice', 'mean'),
        worst_patient_delta_dice=('delta_dice', 'min'),
    )
    summary['operations'] = [operation_count(POLICIES[name]) for name in summary['policy']]
    return summary


def choose(summary, organ):
    rows = summary[summary['organ'] == organ]
    rows = rows[rows['worst_patient_delta_dice'] >= -MAX_DICE_DROP]
    best = rows.loc[rows['mean_delta_hd95_mm'].idxmin()]
    limit = best['mean_delta_hd95_mm'] + best['se_delta_hd95_mm']
    rows = rows[rows['mean_delta_hd95_mm'] <= limit]
    rows = rows.sort_values(['operations', 'mean_delta_hd95_mm'])
    return rows['policy'].iloc[0]


def held_out(results):
    rows = []
    for patient in sorted(results['patient'].unique()):
        summary = summarise(results[results['patient'] != patient])
        test = results[results['patient'] == patient]
        for organ in LABELS:
            policy = choose(summary, organ)
            scored = test[(test['organ'] == organ) & (test['policy'] == policy)]
            rows.append({
                'patient': patient,
                'organ': organ,
                'policy': policy,
                'delta_hd95_mm': scored['delta_hd95_mm'].mean(),
                'delta_dice': scored['delta_dice'].mean(),
            })
    return pd.DataFrame(rows)


def main():
    run_dirs = [RESULTS / f'final_{FINAL_CONFIG}_25ep_s{seed}' for seed in SEEDS]
    test_paths = sorted(TEST_VOLUMES.glob('Patient_*.nii.gz'))
    if not test_paths:
        raise FileNotFoundError(f'No test predictions in {TEST_VOLUMES}')

    rows = []
    for run in run_dirs:
        pred_paths = sorted((run / 'volumes/best_epoch/val').glob('Patient_*.nii.gz'))
        if not pred_paths:
            raise FileNotFoundError(f'No validation predictions in {run / "volumes/best_epoch/val"}')
        for pred_path in pred_paths:
            number = parse_patient_number(pred_path)
            patient = f'Patient_{number:02d}'
            print(f'Evaluating {run.name} {patient}')
            rows += evaluate(run.name, patient, pred_path, patient_volume(GT_ROOT, number))

    results = add_deltas(pd.DataFrame(rows))
    summary = summarise(results)
    policy = {}
    for organ in LABELS:
        policy[organ] = POLICIES[choose(summary, organ)]
    held = held_out(results)

    (DEST / 'volumes').mkdir(parents=True, exist_ok=True)
    results.to_csv(DEST / 'per_patient.csv', index=False)
    summary.to_csv(DEST / 'summary.csv', index=False)
    held.to_csv(DEST / 'held_out.csv', index=False)
    (DEST / 'policy.json').write_text(json.dumps(policy, indent=2) + '\n')
    print(json.dumps(policy, indent=2))
    print(held.groupby('organ')[['delta_hd95_mm', 'delta_dice']].mean())

    for path in test_paths:
        image = nib.load(path)
        pred = np.asarray(image.dataobj, dtype=np.uint8)
        voxel_ml = np.prod(image.header.get_zooms()[:3]) / 1000
        cleaned = apply_policy(pred, policy, voxel_ml)
        nib.save(nib.Nifti1Image(cleaned, image.affine, image.header), DEST / 'volumes' / path.name)
        print(f'{path.name}: postprocessing changed {int((pred != cleaned).sum())} voxels')
    print(f'Saved postprocessed test predictions to {DEST / "volumes"}')


if __name__ == '__main__':
    main()
