#!/usr/bin/env python3
"""Predict SEGTHOR scans, optionally refine organs, and restore NIfTI geometry."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PIL import Image
from torch.utils.data import DataLoader

from dataset import SliceDataset
from main import build_model, img_transform, input_slices
from stitch import stitch_predictions
from utils import tqdm_


def get_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    models = parser.add_mutually_exclusive_group(required=True)
    models.add_argument('--pipeline', type=Path, help='Trusted complete SegThorPipeline pickle.')
    models.add_argument('--weights', type=Path,
                        help='Exact path to a state_dict checkpoint (e.g. bestweights.pt).')
    parser.add_argument('--dest', '--output-dir', dest='dest', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('data/SEGTHOR_TEST'))
    parser.add_argument('--net', choices=['enet', 'enet25d', 'shallow'], default='enet')
    parser.add_argument('--slices', type=int, choices=[3, 5])
    parser.add_argument('--slice-step', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--source-scan-pattern',
                        help='Override original CT paths, e.g. /data/test/{id_}.nii.gz.')
    for organ in ('trachea', 'esophagus', 'aorta', 'heart'):
        parser.add_argument(f'--{organ}-checkpoint', type=Path,
                            help=f'Optional class-bound organ_refine_train.py {organ} checkpoint.')
    parser.add_argument('--refiner-batch-size', type=int, default=16)
    parser.add_argument('--refiner-size', type=int, default=128,
                        help='Crop side used to train the organ refiners (default: 128).')
    args = parser.parse_args(argv)
    if args.pipeline:
        if any(getattr(args, f'{organ}_checkpoint') for organ in ('trachea', 'esophagus', 'aorta', 'heart')):
            parser.error('--pipeline already contains its refiners; omit separate checkpoint flags')
        args.net, args.slices, args.slice_step, args.refiner_size = 'enet25d', 3, 1, 128
    if args.net == 'enet25d' and args.slices is None:
        parser.error('--net enet25d requires --slices 3 or 5')
    if args.net != 'enet25d' and args.slices is not None:
        parser.error('--slices is only used with --net enet25d')
    if args.batch_size < 1 or args.slice_step < 1 or args.num_workers < 0:
        parser.error('batch-size and slice-step must be positive; num-workers must be nonnegative')
    if args.refiner_batch_size < 1 or args.refiner_size < 4:
        parser.error('refiner-batch-size must be positive and refiner-size must be at least 4')
    args.refiner_checkpoints = {class_id: getattr(args, f'{organ}_checkpoint')
                               for class_id, organ in ((3, 'trachea'), (1, 'esophagus'),
                                                       (4, 'aorta'), (2, 'heart'))
                               if getattr(args, f'{organ}_checkpoint') is not None}
    if args.refiner_checkpoints and (args.net != 'enet25d' or args.slices != 3 or args.slice_step != 1):
        parser.error('Phase 7 organ refinement requires --net enet25d --slices 3 --slice-step 1')
    args.dataset = 'SEGTHOR'
    return args


def run_inference(args):
    if getattr(args, 'pipeline', None):
        from segthor_pipeline import load_pipeline
        return load_pipeline(args.pipeline).predict(args)
    metadata_path = args.data_root / 'preprocess.json'
    metadata = json.loads(metadata_path.read_text())
    source_pattern = args.source_scan_pattern or str(
        Path(metadata['source_dir']) / 'test' / '{id_}.nii.gz')
    dataset = SliceDataset('test', args.data_root, img_transform=img_transform,
                           num_classes=5, input_slices=input_slices(args),
                           slice_step=args.slice_step)
    patient_metadata = metadata.get('patients', {}).get('test', {})
    for patient, files in dataset.patients.items():
        if patient not in patient_metadata:
            raise ValueError(f'Missing test preprocessing metadata for {patient}')
        depth = int(patient_metadata[patient]['processed_shape'][2])
        expected = {f'{patient}_{z:04d}' for z in range(depth)}
        if {image.stem for image, _ in files} != expected:
            raise ValueError(f'{patient}: expected all {depth} processed slices starting at zero')
        source = Path(source_pattern.format(id_=patient))
        if not source.is_file():
            raise FileNotFoundError(f'Missing original CT: {source}; use --source-scan-pattern to override')

    device = torch.device(args.device)
    # Match the reported comparison's FP32 teacher and refiner evaluation.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    net = getattr(args, 'embedded_model', None)
    if net is None:
        net = build_model(args, K=5)
        net.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=True))
    net.to(device).eval()
    refiners = getattr(args, 'embedded_refiners', None)
    if refiners:
        from experiments.organ_refine_inference import validate_preprocessing
        validate_preprocessing(metadata, dataset.patients)
        for row in refiners.values():
            row['model'].to(device).eval()
    if getattr(args, 'refiner_checkpoints', {}):
        from experiments.organ_refine_inference import load_refiners, validate_preprocessing
        validate_preprocessing(metadata, dataset.patients)
        refiners = load_refiners(args.refiner_checkpoints, device, coarse_weights=args.weights)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        num_workers=args.num_workers, shuffle=False)
    volumes_dir = args.dest / 'volumes'
    # Avoid mixing predictions from different runs or retaining stale patients.
    if volumes_dir.exists() and any(volumes_dir.iterdir()):
        raise FileExistsError(f'Output folder must be empty: {volumes_dir}')
    volumes_dir.mkdir(parents=True, exist_ok=True)
    refinement_report = None
    if refiners is not None:
        from experiments.organ_refine_inference import run_refined_inference
        paths, refinement_report = run_refined_inference(
            args, dataset, metadata, source_pattern, net, refiners, device, volumes_dir)
        print(f'Saved {len(paths)} refined original-geometry segmentations to {volumes_dir}')
    else:
        # Keep temporary PNGs on the output filesystem (including cluster scratch).
        # The context manager removes them after stitching or on an exception.
        with TemporaryDirectory(prefix='.inference-slices-', dir=args.dest) as temporary:
            slices_dir = Path(temporary)
            with torch.inference_mode():
                for batch in tqdm_(loader):
                    labels = net(batch['images'].to(device)).argmax(dim=1).cpu().numpy()
                    for stem, label in zip(batch['stems'], labels):
                        Image.fromarray(label.astype('uint8')).save(slices_dir / f'{stem}.png')
            paths = stitch_predictions(
                slices_dir, volumes_dir, source_pattern,
                grp_regex=r'(Patient_\d+)_\d+$', num_classes=5,
                preprocess_metadata_path=metadata_path, label_scale=1) # Temporary PNGs contain class IDs 0..4.
        print(f'Saved {len(paths)} original-geometry segmentations to {volumes_dir}')
    digest = hashlib.sha256()
    with args.weights.open('rb') as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b''):
            digest.update(chunk)
    manifest = {
        'status': 'complete', 'postprocessing': 'none',
        'checkpoint_sha256': digest.hexdigest(), 'net': args.net,
        'slices': input_slices(args), 'slice_step': args.slice_step,
        'patients': sorted(dataset.patients),
        'data_root': str(args.data_root.resolve()),
        'source_scan_pattern': source_pattern,
    }
    if refinement_report is not None:
        manifest['refinement'] = refinement_report
    if getattr(args, 'pipeline_sha256', None):
        manifest['pipeline_sha256'] = args.pipeline_sha256
    (args.dest / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return paths


if __name__ == '__main__':
    run_inference(get_args())
