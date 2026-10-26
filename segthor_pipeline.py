"""Portable inference model containing ENet and its matched organ refiners."""
from pathlib import Path

import torch
from torch import nn


class SegThorPipeline(nn.Module):
    """Use predict() for complete scans, which require native CT and geometry."""

    def __init__(self, enet, refiners, coarse_checkpoint):
        super().__init__()
        self.enet = enet
        self.refiners = nn.ModuleDict({str(c): row['model'] for c, row in refiners.items()})
        self.records = {c: {k: v for k, v in row.items() if k != 'model'}
                        for c, row in refiners.items()}
        # Preserve the teacher's exact checkpoint hash, without external paths.
        self.coarse_checkpoint = coarse_checkpoint
        self.config = {'net': 'enet25d', 'slices': 3, 'slice_step': 1, 'refiner_size': 128}

    @classmethod
    def from_checkpoints(cls, model_dir):
        from experiments.trachea_cache_predictions import load_model
        from experiments.organ_refine_inference import load_refiners
        root = Path(model_dir)
        teacher = root / 'enet.pt'
        enet, _ = load_model(teacher, torch.device('cpu'))
        refiners = load_refiners({3: root / 'trachea.pt', 1: root / 'esophagus.pt',
                                 2: root / 'heart.pt'}, torch.device('cpu'), teacher)
        if any(row['coarse_checkpoint_sha256'] is None for row in refiners.values()):
            raise ValueError('Submission refiners must be bound to an ENet checkpoint')
        return cls(enet, refiners, teacher.read_bytes()).cpu().eval()

    def forward(self, images):
        raise RuntimeError('Full refinement requires original CT and geometry; use inference.py --pipeline or predict(args).')

    def predict(self, args):
        """Run the existing native CT inference path using embedded weights."""
        import copy
        import hashlib
        from tempfile import TemporaryDirectory
        from inference import run_inference
        options = copy.copy(args)
        with TemporaryDirectory(prefix='segthor-model-') as temporary:
            teacher = Path(temporary) / 'enet.pt'
            teacher.write_bytes(self.coarse_checkpoint)
            options.weights = teacher
            for key, value in self.config.items():
                setattr(options, key, value)
            options.pipeline = None
            options.refiner_checkpoints = {}
            options.embedded_model = self.enet
            options.embedded_refiners = {
                c: dict(row, model=self.refiners[str(c)]) for c, row in self.records.items()}
            saved_path = getattr(args, 'pipeline', None)
            options.pipeline_sha256 = hashlib.sha256(Path(saved_path).read_bytes()).hexdigest() if saved_path else None
            return run_inference(options)


def load_pipeline(path):
    # Only load trusted project/submission pickle files.
    model = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(model, SegThorPipeline):
        raise ValueError('Expected a saved SegThorPipeline')
    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    model = SegThorPipeline.from_checkpoints(args.model_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, args.output)
