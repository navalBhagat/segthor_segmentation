"""GT-free deployment of the same native crop/refiner recipe used for evaluation."""

from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from experiments.organ_refine_data import ORGANS, compose_crop_logits, crop_patient
from experiments.organ_refine_model import Refiner, canonical_variant, model_logits
from experiments.organ_refine_train import Crops
from experiments.trachea_data import atomic_json, file_sha256, object_sha256
from experiments.trachea_evaluate import restore_hard_label_volume


ORDER = (3, 1, 4, 2)


def validate_preprocessing(metadata, patients):
    """Reject inputs that differ from the phase 7 teacher's preprocessing."""
    if (metadata.get('hu_min') != -1000 or metadata.get('hu_max') != 500
            or metadata.get('shape') != [256, 256]
            or metadata.get('mm_per_px') != 0
            or metadata.get('target_spacing') != [1.0, 1.0, 1.0]):
        raise ValueError('Phase 7 refiners require full-FOV 256x256, HU [-1000,500], '
                         'and target spacing [1,1,1] preprocessing')
    for patient in patients:
        row = metadata['patients']['test'][patient]
        if (row['crop_origin'] != [0, 0] or row['crop_shape'] != row['processed_shape'][:2]
                or row['slice_shape'] != [256, 256]
                or row['processed_spacing'] != [1.0, 1.0, 1.0]):
            raise ValueError(f'{patient}: refinement requires full-FOV phase 7 preprocessing')


def load_refiners(checkpoints, device, coarse_weights=None):
    """Validate organ identity before producing any outputs; retain trained heads."""
    result = {}
    coarse_sha256 = file_sha256(Path(coarse_weights)) if coarse_weights is not None else None
    for class_id in ORDER:
        if class_id not in checkpoints:
            continue
        path = Path(checkpoints[class_id])
        state = torch.load(path, map_location='cpu', weights_only=False)
        config = state['config']
        if (config.get('class_id') != class_id or config.get('input_channels') != 3
                or state.get('completed_epochs', 0) < 1):
            raise ValueError(f'{path}: expected a complete three-CT-channel {ORGANS[class_id]} checkpoint')
        bound_teacher = config.get('validation', {}).get('coarse_checkpoint_sha256')
        if bound_teacher is not None and coarse_sha256 is not None and bound_teacher != coarse_sha256:
            raise ValueError(f'{path}: refiner was trained against a different ENet checkpoint')
        variant = canonical_variant(config['variant'], class_id)
        model = Refiner(config['widths'])
        model.load_state_dict(state['model'], strict=True)
        model.to(device).eval()
        result[class_id] = {'model': model, 'variant': variant,
                            'checkpoint': str(path.resolve()), 'sha256': file_sha256(path),
                            'completed_epochs': state['completed_epochs'],
                            'selection_policy': config.get('selection'),
                            'selection': state.get('selection'),
                            'seed': config.get('seed'), 'coarse_checkpoint_sha256': bound_teacher}
    return result


@torch.inference_mode()
def predict_crop(model, dataset, patient, device, batch_size):
    """Predict combined logits with the training reader's native slice context."""
    result = np.empty(dataset.geometry[patient]['array_shape'], dtype=np.float32)
    for start in range(0, len(dataset), batch_size):
        batch = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        images = torch.stack([item['image'] for item in batch]).to(device)
        base = torch.stack([item['base'] for item in batch]).to(device)
        logits = model_logits(model, images, base)[:, 0].cpu().numpy()
        if not np.isfinite(logits).all():
            raise ValueError(f'{patient}: nonfinite refiner predictions')
        for item, values in zip(batch, logits):
            result[item['z']] = values
    return result


def prepare_patient(args, dataset, indices, patient, metadata, source, net, device,
                    cache, coarse_root, coarse_sha256):
    """Cache hard FP32 decisions and quantized logits exactly as during training."""
    image = nib.load(str(source))
    source_shape = list(image.shape)
    spacing = [float(v) for v in image.header.get_zooms()[:3]]
    if len(source_shape) != 3 or not np.allclose(spacing, metadata['source_spacing'], atol=1e-6, rtol=0):
        raise ValueError(f'{patient}: original CT geometry disagrees with preprocessing metadata')
    expected_processed = [int(round(size * src / dst)) for size, src, dst in
                          zip(source_shape, spacing, metadata['processed_spacing'])]
    if (expected_processed != metadata['processed_shape']
            or metadata.get('source_shape', source_shape) != source_shape
            or not np.allclose(metadata.get('source_affine', image.affine), image.affine, atol=1e-4, rtol=0)):
        raise ValueError(f'{patient}: original CT shape/affine disagrees with preprocessing metadata')
    row = dict(metadata)
    row.update(source_shape=source_shape, source_affine=image.affine.tolist(),
               source_spacing=spacing,
               native_ct_path=str(source.resolve()), image_path=f'{patient}/image.npy')
    row['fingerprint'] = object_sha256({'preprocessing': metadata, 'ct_sha256': file_sha256(source)})
    data_dir, coarse_dir = cache / patient, coarse_root / patient
    data_dir.mkdir(parents=True)
    coarse_dir.mkdir(parents=True)
    shape = (row['processed_shape'][2], *row['slice_shape'])
    prepared = np.lib.format.open_memmap(data_dir / 'image.npy', mode='w+', dtype=np.uint8, shape=shape)
    labels = np.lib.format.open_memmap(coarse_dir / 'prediction.npy', mode='w+', dtype=np.uint8, shape=shape)
    logits = np.lib.format.open_memmap(coarse_dir / 'logits.npy', mode='w+', dtype=np.float16,
                                     shape=(shape[0], 5, *shape[1:]))
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size,
                        num_workers=args.num_workers, shuffle=False)
    with torch.inference_mode():
        for batch in loader:
            values = net(batch['images'].to(device))
            if not torch.isfinite(values).all():
                raise ValueError(f'{patient}: nonfinite ENet predictions')
            # Argmax precedes float16 caching: quantization can change ties.
            hard = values.argmax(dim=1).cpu().numpy().astype(np.uint8)
            quantized = values.cpu().numpy().astype(np.float16)
            if not np.isfinite(quantized).all():
                raise ValueError(f'{patient}: ENet logits overflow the float16 training cache')
            central = batch['images'][:, batch['images'].shape[1] // 2]
            ct = central.mul(255).round().to(torch.uint8).numpy()
            for index, stem in enumerate(batch['stems']):
                z = int(stem.rsplit('_', 1)[1])
                prepared[z], labels[z], logits[z] = ct[index], hard[index], quantized[index]
    prepared.flush()
    labels.flush()
    logits.flush()
    del prepared, labels, logits
    atomic_json(data_dir / 'patient.json', row)
    proof = {'status': 'complete', 'data_fingerprint': row['fingerprint'],
             'checkpoint_sha256': coarse_sha256, 'argmax_dtype': 'float32',
             'logits_dtype': 'float16', 'ground_truth_read': False}
    proof['fingerprint'] = object_sha256(proof)
    atomic_json(coarse_dir / 'patient.json', proof)
    return image, row


def run_refined_inference(args, dataset, metadata, source_pattern, net, refiners, device, volumes_dir):
    """Process one scan at a time, reusing crop preparation and protected composition."""
    positions = {image.stem: index for index, (image, _) in enumerate(dataset.files)}
    coarse_sha256 = file_sha256(args.weights)
    report = {'status': 'running', 'ground_truth_read': False,
              'coarse_checkpoint': str(args.weights.resolve()), 'coarse_sha256': coarse_sha256,
              'order': list(refiners), 'refiners': {
                  ORGANS[class_id]: {key: value for key, value in state.items() if key != 'model'}
                  for class_id, state in refiners.items()}, 'patients': {}}
    atomic_json(args.dest / 'inference.json', report)
    paths = []
    for patient, files in dataset.patients.items():
        source = Path(source_pattern.format(id_=patient))
        with TemporaryDirectory(prefix=f'.refinement-{patient}-', dir=args.dest) as temporary:
            root = Path(temporary)
            cache, coarse = root / 'cache', root / 'coarse'
            image, row = prepare_patient(
                args, dataset, [positions[path.stem] for path, _ in files], patient,
                metadata['patients']['test'][patient], source, net, device, cache, coarse, coarse_sha256)
            low_labels = np.load(coarse / patient / 'prediction.npy', mmap_mode='r', allow_pickle=False)
            current = restore_hard_label_volume(low_labels, row['processed_shape'], row['source_shape'])
            stages = []
            for class_id, state in refiners.items():
                settings = {'class_id': class_id, 'size': args.refiner_size,
                            'minimum_mm': 64. if class_id == 3 else 100., 'margin_mm': 20.,
                            'inference': True, 'window': [-1000, 500]}
                crop_root = root / ORGANS[class_id]
                geometry = crop_patient((cache, coarse, crop_root, patient, settings, float('inf')))
                crops = Crops(crop_root, [patient], class_id, state['variant'], training=False)
                scores = predict_crop(state['model'], crops, patient, device, args.refiner_batch_size)
                candidate = compose_crop_logits(current, scores, geometry['bbox_native_xy'], class_id)
                stages.append({'class_id': class_id, 'bbox_native_xy': geometry['bbox_native_xy'],
                               'placement': geometry['placement'],
                               'changed_voxels': int(np.count_nonzero(current != candidate))})
                current = candidate
                del crops, scores
            header = image.header.copy()
            header.set_data_dtype(np.int16)
            path = volumes_dir / f'{patient}.nii.gz'
            output = nib.Nifti1Image(current.astype(np.int16), image.affine, header=header)
            nib.save(output, str(path))
            report['patients'][patient] = {'source_shape': row['source_shape'], 'stages': stages,
                                            'output': str(path), 'ground_truth_read': False}
            paths.append(path)
        atomic_json(args.dest / 'inference.json', report)
        print(f'{patient}: refined {", ".join(ORGANS[class_id] for class_id in refiners)}', flush=True)
    report['status'] = 'complete'
    atomic_json(args.dest / 'inference.json', report)
    return paths, report
