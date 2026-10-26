#!/usr/bin/env python3

"""Patient-safe training blocks and disjoint axial validation blocks."""

from collections import OrderedDict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dataset import (PairedAugmentation, decode_mask, index_patients,
                     make_dataset, split_patient_stem)
from utils import class2one_hot


class BlockDataset(Dataset):
    def __init__(self, subset, root_dir, depth=16, num_classes=5,
                 augmentation_policy="none", augmentation_components=None,
                 debug=False, img_transform=None, gt_transform=None):
        if depth <= 0:
            raise ValueError("Block depth must be positive")
        if subset != "train" and augmentation_policy != "none":
            raise ValueError("Validation and test blocks cannot be augmented")
        self.depth = depth
        self.num_classes = num_classes
        self.test_mode = subset == "test"
        self.training = subset == "train"
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.paired_transform = PairedAugmentation(augmentation_policy,
                                                  augmentation_components)
        self.augmentation = bool(self.paired_transform.components)

        all_files = make_dataset(root_dir, subset)
        # Build complete volumes before restricting diagnostic center slices.
        self.patients = index_patients(all_files)
        if self.training:
            too_short = [patient for patient, files in self.patients.items()
                         if len(files) < depth]
            if too_short:
                raise ValueError(f"Depth {depth} exceeds training volumes: {too_short}")
        self.positions = {pair[0].stem: (patient, position)
                          for patient, files in self.patients.items()
                          for position, pair in enumerate(files)}
        self.files = all_files[:10] if debug else all_files

        self.blocks = []
        if not self.training:
            selected_patients = index_patients(self.files)
            for patient, files in selected_patients.items():
                for start in range(0, len(files), depth):
                    real_files = files[start:start + depth]
                    self.blocks.append((patient, real_files))

        # Each worker keeps at most about 40 MiB of images/classes at 256x256.
        self.slice_cache = OrderedDict()
        self.cache_size = 128
        print(f">> Created {subset} dataset with {len(self)} blocks, "
              f"{len(self.files)} real center/validation slices and depth {depth}")

    def __len__(self):
        return len(self.files) if self.training else len(self.blocks)

    def _load_slice(self, pair):
        image_path, target_path = pair
        key = image_path.stem
        if key in self.slice_cache:
            self.slice_cache.move_to_end(key)
            return self.slice_cache[key]

        with Image.open(image_path) as image:
            if self.img_transform is None:
                pixels = np.array(image.convert("L"), dtype=np.uint8)
                image_tensor = torch.from_numpy(pixels).float() / 255
            else:
                image_tensor = self.img_transform(image)
                if image_tensor.ndim != 3 or image_tensor.shape[0] != 1:
                    raise ValueError("Block image transform must return [1,H,W]")
                image_tensor = image_tensor[0].float()
        if image_tensor.ndim != 2 or not torch.isfinite(image_tensor).all():
            raise ValueError(f"Invalid image tensor for {image_path}")
        if image_tensor.min() < 0 or image_tensor.max() > 1:
            raise ValueError(f"Image outside [0,1]: {image_path}")

        labels = None
        if target_path is not None:
            with Image.open(target_path) as target:
                labels = decode_mask(target, self.num_classes)
                if self.gt_transform is not None:
                    transformed = self.gt_transform(target)
                    if (transformed.shape != (self.num_classes, *labels.shape)
                            or not torch.all((transformed == 0) | (transformed == 1))
                            or not torch.all(transformed.sum(dim=0) == 1)):
                        raise ValueError("Block target transform must return one-hot [K,H,W]")
                    labels = transformed.argmax(dim=0)
            if labels.shape != image_tensor.shape:
                raise ValueError(f"Image/label shape mismatch for {image_path.stem}")
            labels = labels.to(torch.uint8)

        result = image_tensor, labels
        self.slice_cache[key] = result
        if len(self.slice_cache) > self.cache_size:
            self.slice_cache.popitem(last=False)
        return result

    def __getitem__(self, index):
        if self.training:
            center_path, _ = self.files[index]
            patient, center = self.positions[center_path.stem]
            files = self.patients[patient]
            start = max(0, min(center - self.depth // 2, len(files) - self.depth))
            block_files = files[start:start + self.depth]
        else:
            patient, block_files = self.blocks[index]

        valid_slices = len(block_files)
        padded_files = block_files + [block_files[-1]] * (self.depth - valid_slices)
        loaded = [self._load_slice(pair) for pair in padded_files]
        images = torch.stack([image for image, _ in loaded])
        stems = [pair[0].stem for pair in padded_files]
        assert all(split_patient_stem(stem)[0] == patient for stem in stems)

        result = {"stems": stems, "valid_slices": valid_slices}
        if not self.test_mode:
            labels = torch.stack([target for _, target in loaded]).long()
            if self.training and self.augmentation:
                images, labels = self.paired_transform(images, labels)
            result["gts"] = class2one_hot(labels[None], self.num_classes)[0]
        result["images"] = images[None]
        return result
