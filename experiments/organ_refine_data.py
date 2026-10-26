#!/usr/bin/env python3
"""Prediction-derived native CT crops and protected composition for any organ."""
from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
from scipy.ndimage import generate_binary_structure, label
from skimage.transform import resize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.trachea_data import atomic_json, check_deadline, file_sha256, get_nibabel, load_manifest, object_sha256
from experiments.trachea_evaluate import foreground_bounds, restore_hard_label_volume, restore_scalar_crop

ORGANS = {1: 'esophagus', 2: 'heart', 3: 'trachea', 4: 'aorta'}


def resize_plane(plane, shape, order=1):
    return resize(np.asarray(plane, np.float32), shape, order=order,
                  preserve_range=True, anti_aliasing=False, mode='edge')


def atomic_array(path, values):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.save(stream, values, allow_pickle=False)
    temporary.replace(path)


def coarse_margin(predictions, patient, class_id):
    if class_id not in ORGANS:
        raise ValueError('Expected foreground class 1..4')
    logits = np.load(Path(predictions) / patient / 'logits.npy', mmap_mode='r', allow_pickle=False)
    if logits.ndim != 4 or logits.shape[1] != 5 or not np.isfinite(logits).all():
        raise ValueError('Expected finite Z,5,X,Y coarse logits')
    margin = np.empty((len(logits), *logits.shape[2:]), np.float16)
    others = [index for index in range(5) if index != class_id]
    for z, values in enumerate(logits):
        values = values.astype(np.float32)
        margin[z] = values[class_id] - values[others].max(0)
    if not np.isfinite(margin).all():
        raise ValueError('Nonfinite binary margin')
    return margin


def prediction_crop(prediction, spacing, class_id, minimum_mm=100., margin_mm=20.):
    """One crop rule: robust predicted bounds plus physical context; retain all Z."""
    if class_id not in ORGANS or prediction.ndim != 3 or len(spacing) != 3 or min(spacing) <= 0:
        raise ValueError('Expected native XYZ labels, positive spacing and foreground class')
    mask = prediction == class_id
    status = 'target_quantile_bounds'
    minimum = minimum_mm
    if not mask.any():
        mask = np.isin(prediction, [3, 4]) if class_id == 1 else prediction != 0
        minimum = max(minimum, 128.)
        status = 'predicted_anatomy_fallback'
    coords = np.nonzero(mask)
    crop = []
    for axis in range(2):
        if len(coords[0]):
            lo, hi = np.quantile(coords[axis], [.005, .995])
            center = (lo + hi + 1) / 2
            width = max(hi - lo + 1 + 2 * margin_mm / spacing[axis], minimum / spacing[axis])
        else:
            center = prediction.shape[axis] / 2
            width = max(minimum, 160.) / spacing[axis]
            status = 'image_center_fallback'
        width = min(int(np.ceil(width)), prediction.shape[axis])
        start = max(0, min(int(np.floor(center - width / 2)), prediction.shape[axis] - width))
        crop.append(slice(start, start + width))
    return tuple(crop), {'status': status, 'uses_ground_truth': False, 'class_id': class_id,
                         'quantiles': [.005, .995], 'minimum_mm': minimum_mm, 'margin_mm': margin_mm}


def _centered_interval(center, width, limit):
    width = min(int(width), int(limit))
    start = max(0, min(int(np.floor(center - width / 2)), limit - width))
    return slice(start, start + width)


def component_crop(prediction_zxy, image_zxy, metadata, margin_mm=20.0,
                    minimum_mm=64.0, fallback_mm=128.0):
    """Return two native XY slices and prediction-only placement metadata.

    The largest connected predicted trachea component anchors placement;
    ties follow scipy's first component. A physical margin and minimum field
    protect context. The crop is clipped/shifted within the native CT field.
    """
    native = tuple(int(v) for v in metadata["source_shape"])
    processed = tuple(int(v) for v in metadata["processed_shape"])
    spacing = np.asarray(metadata["source_spacing"], dtype=float)
    if prediction_zxy.ndim != 3 or prediction_zxy.shape != image_zxy.shape:
        raise ValueError("Expected matching coarse Z,X,Y image and prediction")
    if prediction_zxy.shape[0] != processed[2]:
        raise ValueError("Coarse arrays do not cover the processed Z extent")
    if not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("Expected positive finite native spacing")
    if margin_mm < 0 or min(minimum_mm, fallback_mm) <= 0:
        raise ValueError("Use a nonnegative margin and positive crop widths")
    low_mask = prediction_zxy == 3
    bounds = foreground_bounds(low_mask)
    placement = {"uses_gt": False, "margin_mm": float(margin_mm),
                 "minimum_mm": float(minimum_mm), "fallback_mm": float(fallback_mm),
                 "predicted_trachea_voxels_coarse": int(low_mask.sum())}
    if bounds is not None:
        components, count = label(low_mask[bounds], structure=generate_binary_structure(3, 3))
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        anchor_id = int(sizes.argmax())
        low_mask[:] = False
        low_mask[bounds] = components == anchor_id
        del components
        native_anchor = restore_hard_label_volume(low_mask, processed, native)
        native_bounds = foreground_bounds(native_anchor)
        del native_anchor
        # A one-slice coarse component can vanish during axial downsampling.
        # Such a case uses the same explicit empty-anchor fallback below.
        if native_bounds is not None:
            result = []
            for axis in range(2):
                margin = int(np.ceil(margin_mm / spacing[axis]))
                minimum = int(np.ceil(minimum_mm / spacing[axis]))
                width = max(native_bounds[axis].stop - native_bounds[axis].start + 2 * margin,
                            minimum)
                center = (native_bounds[axis].start + native_bounds[axis].stop) / 2
                result.append(_centered_interval(center, width, native[axis]))
            placement.update({"status": "predicted_largest_component", "components": int(count),
                              "anchor_voxels_coarse": int(sizes[anchor_id]),
                              "anchor_bbox_native_xyz": [[s.start, s.stop] for s in native_bounds]})
            return tuple(result), placement
    # Use CT only to locate the body's XY field. Threshold corresponds to
    # approximately -700 HU in the prepared [0,255] image, excluding room air.
    body_xy = np.any(image_zxy > 51, axis=0)
    body_bounds = foreground_bounds(body_xy)
    result = []
    for axis in range(2):
        if body_bounds is None:
            center = native[axis] / 2
        else:
            low_center = (body_bounds[axis].start + body_bounds[axis].stop) / 2
            center = low_center * native[axis] / image_zxy.shape[axis + 1]
        result.append(_centered_interval(center, int(np.ceil(fallback_mm / spacing[axis])),
                                        native[axis]))
    placement.update({"status": "fallback_body_center_no_native_anchor",
                      "warning": "No surviving predicted trachea anchor; coverage requires review"})
    return tuple(result), placement



def compose_crop_logits(baseline, logits, bbox, class_id):
    """Replace only background/target inside the crop; protect all other labels."""
    if class_id not in ORGANS or baseline.ndim != 3 or logits.ndim != 3 or len(logits) != baseline.shape[2]:
        raise ValueError('Expected native XYZ baseline and complete native-Z crop logits')
    if not np.isfinite(logits).all() or not np.isin(baseline, range(5)).all():
        raise ValueError('Invalid labels or logits')
    if len(bbox) != 2 or any(len(b) != 2 or not 0 <= b[0] < b[1] <= baseline.shape[i] for i, b in enumerate(bbox)):
        raise ValueError('Invalid native crop bounds')
    crop = tuple(slice(*bounds) for bounds in bbox)
    result = baseline.copy()
    region = result[crop[0], crop[1], :]
    for z in range(baseline.shape[2]):
        editable = (region[..., z] == 0) | (region[..., z] == class_id)
        decision = resize_plane(logits[z], region.shape[:2]) > 0
        region[..., z][editable] = decision[editable].astype(result.dtype) * class_id
    return result


def crop_patient(task):
    cache, predictions, dest, patient, settings, deadline = task
    cache, predictions, dest = Path(cache), Path(predictions), Path(dest)
    check_deadline(deadline)
    class_id = settings['class_id']
    policy = 'largest_component' if class_id == 3 else 'quantile'
    if settings.get('crop_policy', policy) != policy:
        raise ValueError('Crop policy does not match the selected organ recipe')
    settings = dict(settings, crop_policy=policy)
    if class_id == 3:
        settings['fallback_mm'] = 128.
    meta = json.loads((cache / patient / 'patient.json').read_text())
    pred_meta = json.loads((predictions / patient / 'patient.json').read_text())
    if pred_meta['data_fingerprint'] != meta['fingerprint'] or pred_meta['status'] != 'complete':
        raise ValueError('Coarse predictions do not match source data')
    source = {'data': meta['fingerprint'], 'coarse': pred_meta['fingerprint'], 'class_id': class_id,
              'settings': settings, 'patient': patient,
              'prediction_sha256': file_sha256(predictions / patient / 'prediction.npy'),
              'logits_sha256': file_sha256(predictions / patient / 'logits.npy')}
    if class_id == 3:
        source['prepared_ct_sha256'] = file_sha256(cache / meta['image_path'])
    fingerprint = object_sha256(source)
    directory = dest / patient
    marker = directory / 'geometry.json'
    names = ['ct', 'base'] + ([] if settings['inference'] else ['target', 'valid'])
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous['fingerprint'] != fingerprint:
            raise ValueError('Changed crop inputs; choose a new destination')
        if all((directory / (name + '.npy')).exists() for name in names):
            return previous
    coarse = np.load(predictions / patient / 'prediction.npy', mmap_mode='r', allow_pickle=False)
    if list(coarse.shape) != [meta['processed_shape'][2], *meta['slice_shape']] or not np.isin(coarse, range(5)).all():
        raise ValueError('Invalid coarse prediction geometry or labels')
    native = restore_hard_label_volume(coarse, meta['processed_shape'], meta['source_shape'])
    if class_id == 3:
        if meta['crop_origin'] != [0, 0] or meta['crop_shape'] != meta['processed_shape'][:2]:
            raise ValueError('Component cropping requires full-FOV coarse preprocessing')
        prepared_ct = np.load(cache / meta['image_path'], mmap_mode='r', allow_pickle=False)
        crop, placement = component_crop(coarse, prepared_ct, meta, settings['margin_mm'],
                                         settings['minimum_mm'], settings['fallback_mm'])
        placement['uses_ground_truth'] = False
    else:
        crop, placement = prediction_crop(native, meta['source_spacing'], class_id,
                                          settings['minimum_mm'], settings['margin_mm'])
    nib = get_nibabel()
    image = nib.load(str(cache / meta['native_ct_path']))
    if list(image.shape) != meta['source_shape'] or not np.allclose(image.affine, meta['source_affine']):
        raise ValueError('Native CT geometry mismatch')
    ct = np.asarray(image.dataobj, np.float32)[crop[0], crop[1], :]
    if not np.isfinite(ct).all():
        raise ValueError('Nonfinite native CT values')
    margin = coarse_margin(predictions, patient, class_id)
    base = restore_scalar_crop(margin, meta['processed_shape'], meta['source_shape'], crop, order=1)
    hard_native = (restore_scalar_crop(coarse, meta['processed_shape'], meta['source_shape'], crop, order=0)
                   if class_id == 3 and not settings['inference'] else None)
    truth = None
    if not settings['inference']:
        reference = nib.load(str(cache / meta['native_gt_path']))
        if reference.shape != image.shape or not np.allclose(reference.affine, image.affine):
            raise ValueError('CT and reference geometry differ')
        truth = np.asarray(reference.dataobj, np.uint8)
    shape = (image.shape[2], settings['size'], settings['size'])
    arrays = {name: np.empty(shape, np.uint8 if name in ('target', 'valid') else np.float16) for name in names}
    for z in range(shape[0]):
        check_deadline(deadline)
        normalized = ((np.clip(ct[..., z], -1000., 500.) + 1000.) / 1500. if class_id == 3
                      else np.clip((ct[..., z] + 1000.) / 1500., 0, 1))
        arrays['ct'][z] = resize_plane(normalized, shape[1:])
        arrays['base'][z] = resize_plane(base[z], shape[1:])
        if truth is not None:
            arrays['target'][z] = resize_plane(truth[crop[0], crop[1], z] == class_id, shape[1:], 0) > .5
            if class_id == 3:
                hard = resize_plane(hard_native[z], shape[1:], 0)
                arrays['valid'][z] = (hard == 0) | (hard == class_id)
            else:
                hard = native[crop[0], crop[1], z]
                arrays['valid'][z] = resize_plane((hard == 0) | (hard == class_id), shape[1:], 0) > .5
    geometry = {'patient': patient, 'class_id': class_id, 'fingerprint': fingerprint, 'source': source,
                'array_shape': list(shape), 'array_axes': 'Z,X,Y', 'source_shape': meta['source_shape'],
                'source_spacing': meta['source_spacing'], 'source_affine': meta['source_affine'],
                'bbox_native_xy': [[sl.start, sl.stop] for sl in crop], 'placement': placement,
                'contains_ground_truth': truth is not None, 'settings': settings,
                'coarse_checkpoint_sha256': pred_meta['checkpoint_sha256'],
                'crop_pixel_spacing_mm': [(sl.stop-sl.start)*meta['source_spacing'][i]/settings['size'] for i,sl in enumerate(crop)],
                'coarse_pixel_spacing_mm': [meta['source_shape'][i]*meta['source_spacing'][i]/meta['slice_shape'][i] for i in range(2)]}
    if truth is not None:
        total = int((truth == class_id).sum())
        inside = int((truth[crop[0], crop[1], :] == class_id).sum())
        geometry['diagnostic_only'] = {'reference_voxels': total, 'inside_crop_voxels': inside,
            'crop_coverage': inside / total if total else None,
            'reference_voxels_editable': int(((truth == class_id) & np.isin(native, [0, class_id])).sum())}
    directory.mkdir(parents=True, exist_ok=True)
    for name, values in arrays.items():
        atomic_array(directory / (name + '.npy'), values)
    atomic_json(marker, geometry)
    return geometry


def main(args):
    manifest = load_manifest(args.cache)
    patients = [f'Patient_{int(p):02d}' if p.isdigit() else p for p in args.patients]
    if not patients or len(set(patients)) != len(patients) or set(patients) - set(manifest['patients']):
        raise ValueError('Expected unique known patients')
    settings = {'class_id': args.class_id, 'size': args.size,
                'minimum_mm': args.minimum_mm if args.minimum_mm is not None else (64. if args.class_id == 3 else 100.),
                'crop_policy': 'largest_component' if args.class_id == 3 else 'quantile',
                'margin_mm': args.margin_mm, 'inference': args.inference, 'window': [-1000, 500],
                'source_sha256': file_sha256(Path(__file__))}
    if args.class_id == 3:
        settings['fallback_mm'] = 128.
    args.dest.mkdir(parents=True, exist_ok=True)
    if (args.dest / 'manifest.json').exists():
        previous = json.loads((args.dest / 'manifest.json').read_text())
        if previous['settings'] != settings:
            raise ValueError('Changed crop settings; choose a new destination')
    report = {'status': 'running', 'class_id': args.class_id, 'settings': settings, 'patients': {}}
    deadline = time.monotonic() + args.max_seconds
    tasks = [(str(args.cache), str(args.predictions), str(args.dest), p, settings, deadline) for p in patients]
    try:
        with multiprocessing.get_context('spawn').Pool(args.workers) as pool:
            for geometry in pool.imap_unordered(crop_patient, tasks):
                report['patients'][geometry['patient']] = geometry
                atomic_json(args.dest / 'manifest.json', report)
                print(json.dumps({'patient': geometry['patient'], 'class_id': args.class_id,
                                  'diagnostic': geometry.get('diagnostic_only')}), flush=True)
        report['status'] = 'complete'
    except Exception as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        atomic_json(args.dest / 'manifest.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--dest', type=Path, required=True)
    parser.add_argument('--patients', nargs='+', required=True)
    parser.add_argument('--class-id', type=int, choices=ORGANS, required=True)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--minimum-mm', type=float, help='Defaults to 64 mm for trachea, 100 mm for other organs')
    parser.add_argument('--margin-mm', type=float, default=20.)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--max-seconds', type=float, default=1140)
    parser.add_argument('--inference', action='store_true')
    options = parser.parse_args()
    if options.size < 4 or options.workers < 1 or (options.minimum_mm is not None and options.minimum_mm <= 0) or options.margin_mm < 0 or options.max_seconds <= 0:
        parser.error('Invalid crop settings')
    main(options)
