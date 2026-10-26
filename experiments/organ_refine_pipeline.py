"""Validate and compose cached organ predictions in native scan geometry.

The Phase 7 experiment driver uses these helpers for protected composition;
reported metrics are computed by the repository's shared metrics_3d module.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from experiments.organ_refine_data import compose_crop_logits
from experiments.trachea_data import check_deadline, file_sha256, get_nibabel, object_sha256
from experiments.trachea_evaluate import restore_hard_label_volume

ORDER = (3, 1, 4, 2)


def read_json(path):
    return json.loads(Path(path).read_text())


def signature(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'sha256': file_sha256(path)}


def validate_stages(stages):
    classes = [stage['class_id'] for stage in stages]
    if not classes or len(set(classes)) != len(classes) or classes != [c for c in ORDER if c in classes]:
        raise ValueError('Stages must be a distinct nonempty subset in order 3,1,4,2')
    for stage in stages:
        if stage.get('kind', 'logits') != 'logits':
            raise ValueError('Unknown stage kind')
        if not all(stage.get(key) for key in ('crops', 'predictions')):
            raise ValueError('Every stage requires crops and predictions')
    return stages


def check_geometry(geometry, metadata):
    if (geometry['source_shape'] != metadata['source_shape']
            or not np.allclose(geometry['source_spacing'], metadata['source_spacing'], atol=1e-6, rtol=0)
            or not np.allclose(geometry['source_affine'], metadata['source_affine'], atol=1e-4, rtol=0)
            or geometry['array_shape'][0] != metadata['source_shape'][2]):
        raise ValueError('Stage native geometry mismatch')
    if geometry.get('fingerprint') != object_sha256(geometry['source']):
        raise ValueError('Crop fingerprint changed')
    placement = geometry['placement']
    if placement.get('uses_ground_truth', placement.get('uses_gt')) is not False:
        raise ValueError('Crop placement must explicitly exclude ground truth')


def validate_coarse(proof, metadata):
    if (proof.get('status') != 'complete' or proof.get('data_fingerprint') != metadata['fingerprint']
            or not proof.get('checkpoint_sha256')):
        raise ValueError('Incomplete coarse prediction or wrong data cache')


def stage_scores(stage, patient, metadata, coarse_proof, coarse_hashes):
    """Read predictions and provenance only; never CT targets or GT volumes."""
    class_id = stage['class_id']
    crop_dir = Path(stage['crops']) / patient
    pred_root = Path(stage['predictions'])
    pred_dir = pred_root / patient
    geom_path, proof_path = crop_dir / 'geometry.json', pred_dir / 'patient.json'
    geometry, proof = read_json(geom_path), read_json(proof_path)
    check_geometry(geometry, metadata)
    if proof.get('patient') != patient or geometry.get('patient') != patient:
        raise ValueError('Stage patient mismatch')
    if proof.get('crop_fingerprint') != geometry['fingerprint']:
        raise ValueError('Prediction belongs to another crop')
    if proof.get('ground_truth_read') is not False or proof.get('target_files_read') is not False:
        raise ValueError('Prediction requires explicit GT-free provenance')
    source = geometry['source']
    paths = [geom_path, proof_path]
    global_path = pred_root / 'prediction.json'
    global_proof = read_json(global_path)
    paths.append(global_path)
    if (global_proof.get('class_id') != class_id or proof.get('class_id') != class_id
            or not proof.get('checkpoint_sha256')
            or global_proof.get('checkpoint_sha256') != proof['checkpoint_sha256']):
        raise ValueError('Logit class/checkpoint provenance mismatch')
    if source.get('data') != metadata['fingerprint'] or geometry.get('coarse_checkpoint_sha256') != coarse_proof['checkpoint_sha256']:
        raise ValueError('Stage uses a different coarse seed or cache')
    if any(value != class_id for value in (geometry.get('class_id'), source.get('class_id'), geometry['settings'].get('class_id'))):
        raise ValueError('Generic crop class mismatch')
    if source.get('coarse') != coarse_proof['fingerprint']:
        raise ValueError('Generic crop teacher mismatch')
    for name in ('prediction', 'logits'):
        if source.get(name + '_sha256') != coarse_hashes[name]:
            raise ValueError('Generic crop coarse array mismatch')
    binding = global_proof.get('data', {}).get(patient, {})
    if binding.get('geometry') != file_sha256(geom_path) or binding.get('crop_fingerprint') != geometry['fingerprint']:
        raise ValueError('Global prediction crop binding mismatch')
    output = pred_dir / 'logits.npy'
    if proof.get('logits_sha256') != file_sha256(output):
        raise ValueError('Logits changed since prediction')
    scores = np.load(output, mmap_mode='r', allow_pickle=False)
    if scores.dtype != np.float32 or not np.isfinite(scores).all():
        raise ValueError('Expected finite float32 scores')
    if list(scores.shape) != geometry['array_shape']:
        raise ValueError('Logits do not match crop shape')
    paths.append(output)
    record = {'class_id': class_id, 'kind': 'logits',
        'checkpoint_sha256': proof['checkpoint_sha256'], 'crop_fingerprint': geometry['fingerprint'],
        'files': [signature(path) for path in paths], 'ground_truth_read': False}
    return scores, geometry, record


def compose_patient(patient, metadata, coarse_root, stages, deadline):
    """Return original and final native labels without reading any GT file."""
    validate_stages(stages)
    coarse_root = Path(coarse_root)
    coarse_dir = coarse_root / patient
    proof = read_json(coarse_dir / 'patient.json')
    validate_coarse(proof, metadata)
    hashes = {name: file_sha256(coarse_dir / (name + '.npy')) for name in ('prediction', 'logits')}
    coarse = np.load(coarse_dir / 'prediction.npy', mmap_mode='r', allow_pickle=False)
    if list(coarse.shape) != [metadata['processed_shape'][2], *metadata['slice_shape']] or not np.isin(coarse, range(5)).all():
        raise ValueError('Common prediction geometry or labels invalid')
    full_logits = np.load(coarse_dir / 'logits.npy', mmap_mode='r', allow_pickle=False)
    if list(full_logits.shape) != [coarse.shape[0], 5, *coarse.shape[1:]]:
        raise ValueError('Common logits must contain all five classes at the coarse geometry')
    common = restore_hard_label_volume(coarse, metadata['processed_shape'], metadata['source_shape'])
    current = common.copy()
    records = []
    for stage in stages:
        check_deadline(deadline)
        scores, geometry, record = stage_scores(stage, patient, metadata, proof, hashes)
        candidate = compose_crop_logits(current, scores, geometry['bbox_native_xy'], stage['class_id'])
        protected = ~np.isin(current, [0, stage['class_id']])
        if np.any(candidate[protected] != current[protected]):
            raise AssertionError('Pipeline stage changed another current organ')
        record['changed_voxels'] = int(np.count_nonzero(candidate != current))
        record['protected_changed_voxels'] = 0
        records.append(record)
        current = candidate
    return common, current, {'coarse_checkpoint_sha256': proof['checkpoint_sha256'],
        'coarse_files': [signature(coarse_dir / name) for name in ('patient.json', 'prediction.npy', 'logits.npy')],
        'stages': records, 'composition_ground_truth_read': False}


def load_native(path, metadata):
    image = get_nibabel().load(str(path))
    if list(image.shape) != metadata['source_shape'] or not np.allclose(image.affine, metadata['source_affine'], atol=1e-4, rtol=0):
        raise ValueError('Comparison/reference native geometry mismatch')
    values = np.asarray(image.dataobj)
    if not np.isin(values, range(5)).all():
        raise ValueError('Invalid native labels')
    return values.astype(np.uint8)


def save_native(path, volume, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name.replace('.nii.gz', '.tmp.nii.gz'))
    get_nibabel().save(get_nibabel().Nifti1Image(volume.astype(np.uint8), np.asarray(metadata['source_affine'])), str(temporary))
    temporary.replace(path)
