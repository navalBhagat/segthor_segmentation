#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from pathlib import Path
from typing import Callable, Union
import re

import numpy as np
import torch
from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from utils import class2one_hot


AUGMENTATION_POLICIES = {
    'none': {},
    'mild': {
        'affine': {'probability': 0.5, 'degrees': 10, 'translate': 0.05,
                   'scale': (0.9, 1.1)},
        'gamma': {'probability': 0.3, 'range': (0.9, 1.1)},
        'noise': {'probability': 0.2, 'std': (0.0, 0.02)},
    },
    'strong': {
        'affine': {'probability': 0.7, 'degrees': 15, 'translate': 0.1,
                   'scale': (0.8, 1.2)},
        'gamma': {'probability': 0.3, 'range': (0.7, 1.5)},
        'noise': {'probability': 0.2, 'std': (0.0, 0.04)},
        'contrast': {'probability': 0.3, 'range': (0.8, 1.2)},
        'blur': {'probability': 0.2, 'sigma': (0.5, 1.0), 'kernel_size': 7},
    },
}


def make_dataset(root, subset) -> list[tuple[Path, Path | None]]:
    assert subset in ['train', 'val', 'test']
    root = Path(root)
    print(f"> {root=}")
    images = sorted((root / subset / 'img').glob('*.png'))
    if not images:
        raise ValueError(f'No PNG images found in {root / subset / "img"}')
    if subset == 'test':
        return [(path, None) for path in images]

    labels = {path.stem: path for path in (root / subset / 'gt').glob('*.png')}
    image_stems = {path.stem for path in images}
    if image_stems != labels.keys():
        missing = sorted(image_stems - labels.keys())
        extra = sorted(labels.keys() - image_stems)
        raise ValueError(f'Image/mask stems differ: missing masks {missing[:5]}, '
                         f'extra masks {extra[:5]}')
    return [(path, labels[path.stem]) for path in images]


def split_patient_stem(stem: str) -> tuple[str, int]:
    match = re.fullmatch(r'(Patient_\d+)_(\d+)', stem)
    if match is None:
        raise ValueError(f'Expected Patient_ID_slice filename, got {stem!r}')
    return match.group(1), int(match.group(2))


def index_patients(files) -> dict[str, list[tuple[Path, Path | None]]]:
    patients = {}
    for image, label in files:
        patient, number = split_patient_stem(image.stem)
        volume = patients.setdefault(patient, {})
        if number in volume:
            raise ValueError(f'Duplicate slice number for {patient}: {number}')
        volume[number] = (image, label)

    result = {}
    for patient, volume in sorted(patients.items()):
        numbers = sorted(volume)
        if numbers != list(range(numbers[0], numbers[-1] + 1)):
            raise ValueError(f'Missing interior slices in {patient}')
        result[patient] = [volume[number] for number in numbers]
    return result


def decode_mask(mask: Image.Image, num_classes: int) -> Tensor:
    if num_classes < 2:
        raise ValueError('A segmentation mask needs at least two classes')
    values = np.asarray(mask)
    if values.ndim != 2:
        raise ValueError(f'Expected grayscale mask, got shape {values.shape}')
    step = 63 if num_classes == 5 else 255 / (num_classes - 1)
    allowed = (np.arange(num_classes) * step).astype(np.uint8)
    unexpected = np.setdiff1d(np.unique(values), allowed)
    if unexpected.size:
        raise ValueError(f'Unexpected mask bytes {unexpected.tolist()}; '
                         f'expected {allowed.tolist()}')
    # A lookup retains exact integer labels, including quantized byte encodings.
    lookup = np.zeros(256, dtype=np.int64)
    lookup[allowed] = np.arange(num_classes)
    return torch.from_numpy(lookup[values])


class PairedAugmentation:
    def __init__(self, policy='mild', components=None):
        if policy not in AUGMENTATION_POLICIES:
            raise ValueError(f'Unknown augmentation policy: {policy}')
        self.policy = policy
        settings = AUGMENTATION_POLICIES[policy]
        self.components = tuple(settings if components is None else components)
        if len(set(self.components)) != len(self.components):
            raise ValueError('Augmentation components must not be repeated')
        unknown = set(self.components) - settings.keys()
        if unknown:
            raise ValueError(f'Components {sorted(unknown)} are unavailable in {policy}')
        self.settings = {name: dict(settings[name]) for name in self.components}

    def __call__(self, images: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        if images.shape[-2:] != labels.shape[-2:]:
            raise ValueError('Image and mask spatial shapes differ')
        if not self.components:
            return images, labels
        if images.device.type != 'cpu' or labels.device.type != 'cpu':
            raise ValueError('Online augmentation runs in CPU dataset workers')
        if labels.is_floating_point():
            raise ValueError('Decode masks to integer labels before augmentation')

        # One worker-seeded draw per sample keeps ablation randomness comparable.
        generator = torch.Generator().manual_seed(torch.randint(2**31 - 1, ()).item())
        draws = torch.rand(13, generator=generator).tolist()
        gates = dict(zip(('affine', 'gamma', 'noise', 'contrast', 'blur'), draws[:5]))
        def active(name):
            return name in self.settings and gates[name] < self.settings[name]['probability']
        def interpolate(bounds, draw):
            return bounds[0] + draw * (bounds[1] - bounds[0])

        image_shape, label_shape = images.shape, labels.shape
        height, width = image_shape[-2:]
        planes = images.reshape(-1, height, width)
        masks = labels.reshape(-1, height, width)
        if active('affine'):
            setting = self.settings['affine']
            angle = (2 * draws[5] - 1) * setting['degrees']
            shift = [round((2 * draws[6] - 1) * setting['translate'] * width),
                     round((2 * draws[7] - 1) * setting['translate'] * height)]
            scale = interpolate(setting['scale'], draws[8])
            planes = TF.affine(planes, angle, shift, scale, [0.0, 0.0],
                               interpolation=InterpolationMode.BILINEAR, fill=0)
            masks = TF.affine(masks.float(), angle, shift, scale, [0.0, 0.0],
                              interpolation=InterpolationMode.NEAREST, fill=0).to(labels.dtype)
        if active('gamma'):
            planes = planes.pow(interpolate(self.settings['gamma']['range'], draws[9]))
        if active('noise'):
            std = interpolate(self.settings['noise']['std'], draws[10])
            planes = (planes + torch.randn(planes.shape, generator=generator,
                                            dtype=planes.dtype) * std).clamp(0, 1)
        if active('contrast'):
            factor = interpolate(self.settings['contrast']['range'], draws[11])
            mean = planes.mean()
            planes = ((planes - mean) * factor + mean).clamp(0, 1)
        if active('blur'):
            setting = self.settings['blur']
            sigma = interpolate(setting['sigma'], draws[12])
            planes = TF.gaussian_blur(planes, setting['kernel_size'], [sigma, sigma])
        return planes.clamp(0, 1).reshape(image_shape), masks.reshape(label_shape)


class SliceDataset(Dataset):
    def __init__(self, subset, root_dir, img_transform=None,
                 gt_transform=None, augment=False, equalize=False, debug=False,
                 input_slices=1, slice_step=1, num_classes=None,
                 augmentation_policy='none', augmentation_components=None):
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.equalize: bool = equalize
        self.test_mode: bool = subset == 'test'
        self.num_classes = num_classes
        self.input_slices = input_slices
        self.slice_step = slice_step
        if input_slices not in (1, 3, 5) or not isinstance(slice_step, int) or slice_step < 1:
            raise ValueError('Use 1, 3 or 5 input slices and a positive integer slice step')
        if augment and augmentation_policy == 'none':
            augmentation_policy = 'mild'
        if subset != 'train' and augmentation_policy != 'none':
            raise ValueError('Validation/test augmentation is disabled')
        self.paired_transform = PairedAugmentation(augmentation_policy, augmentation_components)
        self.augmentation = bool(self.paired_transform.components)
        if (input_slices != 1 or slice_step != 1 or self.augmentation) and num_classes != 5:
            raise ValueError('Context and augmentation require SEGTHOR with num_classes=5')

        all_files = make_dataset(root_dir, subset)
        self.patients = index_patients(all_files) if num_classes == 5 else {}
        self.positions = {image.stem: (patient, index)
                          for patient, files in self.patients.items()
                          for index, (image, _) in enumerate(files)}
        # Neighbour lookup always uses the complete split, including in debug mode.
        self.files = all_files[:10] if debug else all_files
        print(f">> Created {subset} dataset with {len(self)} images...")

    def __len__(self):
        return len(self.files)

    def context_files(self, index) -> list[tuple[Path, Path | None]]:
        image, label = self.files[index]
        if self.input_slices == 1:
            return [(image, label)]
        patient, center = self.positions[image.stem]
        volume = self.patients[patient]
        radius = self.input_slices // 2
        return [volume[max(0, min(len(volume) - 1, center + offset * self.slice_step))]
                for offset in range(-radius, radius + 1)]

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]
        images = []
        for path, _ in self.context_files(index):
            with Image.open(path) as image:
                images.append(self.img_transform(image))
        img = torch.cat(images, dim=0)
        if img.ndim != 3 or img.shape[0] != self.input_slices:
            raise ValueError(f'Expected {self.input_slices} grayscale channels, got {img.shape}')
        data_dict = {'images': img, 'stems': img_path.stem}

        if not self.test_mode:
            with Image.open(gt_path) as mask:
                if self.num_classes == 5:
                    labels = decode_mask(mask, self.num_classes)
                else:
                    gt = self.gt_transform(mask)
            if self.num_classes == 5:
                if self.augmentation:
                    img, labels = self.paired_transform(img, labels)
                gt = class2one_hot(labels[None], K=self.num_classes)[0]
                data_dict['images'] = img
            assert gt.shape[1:] == img.shape[1:]
            data_dict['gts'] = gt
        return data_dict
