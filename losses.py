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


import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch import Tensor, einsum

from utils import simplex, sset


LOSS_NAMES = (
    'ce',
    'w_ce',
    'dice',
    'w_dice',
    'ce_dice',
    'w_ce_dice',
    'ce_w_dice',
    'ce_sqrt_w_dice',
    'w_ce_w_dice',
    'boundary',
    'ce_boundary',
    'w_ce_boundary',
    'ce_dice_boundary',
    'w_ce_dice_boundary',
    'ce_w_dice_boundary',
    'ce_sqrt_w_dice_boundary',
    'w_ce_w_dice_boundary',
)


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def _log_probabilities(self, pred_softmax: Tensor, logits: Tensor | None) -> Tensor:
        if logits is not None:
            assert logits.shape == pred_softmax.shape
            return logits.float().log_softmax(dim=1)[:, self.idk, ...]
        # Probability-only callers remain supported; training supplies logits to avoid underflow.
        probs = pred_softmax[:, self.idk, ...].float()
        return probs.clamp_min(torch.finfo(probs.dtype).tiny).log()

    def __call__(self, pred_softmax, weak_target, *, logits=None):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = self._log_probabilities(pred_softmax, logits)
        mask = weak_target[:, self.idk, ...].float()

        loss = - einsum("bk...,bk...->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class WeightedCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = kwargs.get('class_weights', kwargs.get('ce_class_weights'))
        self.class_weight_power = kwargs.get('class_weight_power', 0.5)

    def _weights(self, mask: Tensor) -> Tensor:
        if self.class_weights is not None:
            weights = torch.as_tensor(self.class_weights, device=mask.device, dtype=mask.dtype)
            return weights[self.idk] if weights.numel() != len(self.idk) else weights

        class_sizes = einsum("bk...->k", mask)
        present = class_sizes > 0
        weights = torch.zeros_like(class_sizes)
        weights[present] = 1 / (class_sizes[present] + 1e-10).pow(self.class_weight_power)
        weights[present] /= weights[present].mean().clamp_min(1e-10)
        return weights

    def __call__(self, pred_softmax, weak_target, *, logits=None):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = self._log_probabilities(pred_softmax, logits)
        mask = weak_target[:, self.idk, ...].float()
        weights = self._weights(mask).view(1, -1, *([1] * (mask.ndim - 2)))
        weighted_mask = weights * mask

        loss = - einsum("bk...,bk...->", weighted_mask, log_p)
        loss /= weighted_mask.sum() + 1e-10

        return loss


class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)
        

class DiceLoss(): 
    def __init__(self,**kwargs): 
        self.idk = kwargs['idk']
        self.smooth = kwargs.get('smooth', 1e-6)
        self.generalized = kwargs.get('generalized', False)  # toggle weighting
        self.weight_power = kwargs.get('dice_weight_power', 2.0)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")
        
    def __call__(self, pred_softmax, weak_target, *, logits=None):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        #==> Compute DICE over foreground
        foreground_idk = [k for k in self.idk if k != 0]

        intersection = einsum("bk...,bk...->bk", pred_softmax[:, foreground_idk, ...], weak_target[:, foreground_idk, ...].float())
        union = einsum("bk...->bk", pred_softmax[:, foreground_idk, ...]) + einsum("bk...->bk", weak_target[:, foreground_idk, ...].float())
        
        if self.generalized:
            class_sizes = einsum("bk...->bk", weak_target[:, foreground_idk, ...].float())
            present = class_sizes > 0
            w = class_sizes.clamp_min(1).pow(-self.weight_power) * present
            w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-10)
            # Reduce classes before division; per-class ratios cancel the weights.
            intersection = (w * intersection).sum(dim=1)
            union = (w * union).sum(dim=1)
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            valid = present.any(dim=1)
            return ((1 - dice) * valid).sum() / valid.sum().clamp_min(1)
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice.mean()  # Average over classes and batch
        
        return loss
    

class CrossEntropyDiceLoss(): 
    def __init__(self, **kwargs): 
        ce_loss_class = kwargs.get('ce_loss_class', CrossEntropy)
        ce_kwargs = {k: v for k, v in kwargs.items() if k != 'ce_loss_class'}
        self.ce_loss = ce_loss_class(**ce_kwargs)

        # Dice only on foreground classes
        foreground_kwargs = {**ce_kwargs, 'idk': [k for k in kwargs['idk'] if k != 0]}
        self.dice_loss = DiceLoss(**foreground_kwargs)
        self.ce_weight = kwargs.get('ce_weight', 1.0)
        self.dice_weight = kwargs.get('dice_weight', 1.0)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")
    
    def __call__(self, pred_softmax, weak_target, *, logits=None):
        return (self.ce_weight * self.ce_loss(pred_softmax, weak_target, logits=logits)
                + self.dice_weight * self.dice_loss(pred_softmax, weak_target))


def _one_hot_to_signed_distance(one_hot: np.ndarray) -> np.ndarray:
    distances = np.zeros(one_hot.shape, dtype=np.float32)
    for class_index, mask in enumerate(one_hot.astype(bool)):
        if not mask.any():
            continue
        background = ~mask
        distances[class_index] = (
            distance_transform_edt(background) * background
            - (distance_transform_edt(mask) - 1) * mask
        )
    return distances


class BoundaryLoss():
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax: Tensor, weak_target: Tensor, *, logits=None) -> Tensor:
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        foreground_idk = [k for k in self.idk if k != 0]
        target = weak_target[:, foreground_idk, ...].detach().cpu().numpy()
        dist_maps = np.stack([_one_hot_to_signed_distance(sample) for sample in target], axis=0)
        dist_maps = torch.as_tensor(dist_maps, device=pred_softmax.device, dtype=pred_softmax.dtype)
        probs = pred_softmax[:, foreground_idk, ...].float()
        return einsum("bk...,bk...->", probs, dist_maps) / probs.numel()


class CrossEntropyBoundaryLoss():
    def __init__(self, **kwargs):
        ce_loss_class = kwargs.get('ce_loss_class', CrossEntropy)
        ce_kwargs = {k: v for k, v in kwargs.items() if k != 'ce_loss_class'}
        self.ce_loss = ce_loss_class(**ce_kwargs)
        foreground_kwargs = {**ce_kwargs, 'idk': [k for k in kwargs['idk'] if k != 0]}
        self.boundary_loss = BoundaryLoss(**foreground_kwargs)
        self.ce_weight = kwargs.get('ce_weight', 1.0)
        self.boundary_weight = kwargs.get('boundary_weight', 0.01)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax: Tensor, weak_target: Tensor, *, logits=None) -> Tensor:
        return (self.ce_weight * self.ce_loss(pred_softmax, weak_target, logits=logits)
                + self.boundary_weight * self.boundary_loss(pred_softmax, weak_target))


class CrossEntropyDiceBoundaryLoss():
    def __init__(self, **kwargs):
        ce_loss_class = kwargs.get('ce_loss_class', CrossEntropy)
        ce_kwargs = {k: v for k, v in kwargs.items() if k != 'ce_loss_class'}
        self.ce_loss = ce_loss_class(**ce_kwargs)
        foreground_kwargs = {**ce_kwargs, 'idk': [k for k in kwargs['idk'] if k != 0]}
        self.dice_loss = DiceLoss(**foreground_kwargs)
        self.boundary_loss = BoundaryLoss(**foreground_kwargs)
        self.ce_weight = kwargs.get('ce_weight', 1.0)
        self.dice_weight = kwargs.get('dice_weight', 1.0)
        self.boundary_weight = kwargs.get('boundary_weight', 0.01)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax: Tensor, weak_target: Tensor, *, logits=None) -> Tensor:
        return (self.ce_weight * self.ce_loss(pred_softmax, weak_target, logits=logits)
                + self.dice_weight * self.dice_loss(pred_softmax, weak_target)
                + self.boundary_weight * self.boundary_loss(pred_softmax, weak_target))


def get_loss(loss_name: str, idk: list[int], **kwargs):
    loss_kwargs = {'idk': idk, **kwargs}
    if loss_name == 'ce':
        return CrossEntropy(**loss_kwargs)
    if loss_name == 'w_ce':
        return WeightedCrossEntropy(**loss_kwargs)
    if loss_name == 'dice':
        return DiceLoss(**loss_kwargs)
    if loss_name == 'w_dice':
        return DiceLoss(**loss_kwargs, generalized=True)
    if loss_name == 'ce_dice':
        return CrossEntropyDiceLoss(**loss_kwargs)
    if loss_name == 'w_ce_dice':
        return CrossEntropyDiceLoss(**loss_kwargs, ce_loss_class=WeightedCrossEntropy)
    if loss_name == 'ce_w_dice':
        return CrossEntropyDiceLoss(**loss_kwargs, generalized=True)
    if loss_name == 'ce_sqrt_w_dice':
        return CrossEntropyDiceLoss(**loss_kwargs, generalized=True, dice_weight_power=0.5)
    if loss_name == 'w_ce_w_dice':
        return CrossEntropyDiceLoss(**loss_kwargs, generalized=True, ce_loss_class=WeightedCrossEntropy)
    if loss_name == 'boundary':
        return BoundaryLoss(**loss_kwargs)
    if loss_name == 'ce_boundary':
        return CrossEntropyBoundaryLoss(**loss_kwargs)
    if loss_name == 'w_ce_boundary':
        return CrossEntropyBoundaryLoss(**loss_kwargs, ce_loss_class=WeightedCrossEntropy)
    if loss_name == 'ce_dice_boundary':
        return CrossEntropyDiceBoundaryLoss(**loss_kwargs)
    if loss_name == 'w_ce_dice_boundary':
        return CrossEntropyDiceBoundaryLoss(**loss_kwargs, ce_loss_class=WeightedCrossEntropy)
    if loss_name == 'ce_w_dice_boundary':
        return CrossEntropyDiceBoundaryLoss(**loss_kwargs, generalized=True)
    if loss_name == 'ce_sqrt_w_dice_boundary':
        return CrossEntropyDiceBoundaryLoss(**loss_kwargs, generalized=True, dice_weight_power=0.5)
    if loss_name == 'w_ce_w_dice_boundary':
        return CrossEntropyDiceBoundaryLoss(**loss_kwargs, generalized=True, ce_loss_class=WeightedCrossEntropy)
    raise ValueError(f"Unknown loss: {loss_name}")
    
