"""Shared binary residual refiner; organ identity belongs to data provenance."""
import torch
from torch import nn
from torch.nn import functional as F

VARIANTS = ("per_image", "batch")
PRIOR_CLIP = 4.0


def canonical_variant(variant, class_id):
    if class_id not in (1, 2, 3, 4) or isinstance(class_id, bool):
        raise ValueError("class_id must be one of 1, 2, 3, 4")
    if variant not in VARIANTS:
        raise ValueError(f"Unknown refiner variant: {variant}")
    return variant


class _ConvBlock(nn.Sequential):
    def __init__(self, inputs, outputs):
        groups = min(4, outputs)
        while outputs % groups:
            groups -= 1
        super().__init__(
            nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
            nn.GroupNorm(groups, outputs), nn.SiLU(inplace=True),
            nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
            nn.GroupNorm(groups, outputs), nn.SiLU(inplace=True))


class Refiner(nn.Module):
    """Three CT slices predict an unbounded correction to a clipped prior.

    Parameter names/shapes match the selected three-channel EsophagusNet.
    Fresh training zeroes the head; loading a checkpoint never resets it.
    """
    def __init__(self, widths=(8, 16, 32)):
        super().__init__()
        if len(widths) != 3 or min(widths) < 1:
            raise ValueError("Use three positive widths")
        first, second, third = widths
        self.encoder1 = _ConvBlock(3, first)
        self.encoder2 = _ConvBlock(first, second)
        self.bottleneck = _ConvBlock(second, third)
        self.decoder2 = _ConvBlock(third + second, second)
        self.decoder1 = _ConvBlock(second + first, first)
        self.head = nn.Conv2d(first, 1, 1)

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3 or min(images.shape[-2:]) < 4:
            raise ValueError("Expected B,3,H,W images with H,W >= 4")
        first = self.encoder1(images)
        second = self.encoder2(F.avg_pool2d(first, 2))
        deep = self.bottleneck(F.avg_pool2d(second, 2))
        deep = F.interpolate(deep, size=second.shape[-2:], mode="bilinear", align_corners=False)
        up = self.decoder2(torch.cat((deep, second), dim=1))
        up = F.interpolate(up, size=first.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.decoder1(torch.cat((up, first), dim=1)))


def model_logits(model, images, base):
    correction = model(images)
    if base.shape != correction.shape:
        raise ValueError("Expected matching B,1,H,W coarse margins")
    return base.to(correction).clamp(-PRIOR_CLIP, PRIOR_CLIP) + correction


def refiner_loss(logits, target, valid, variant="batch"):
    """Half masked BCE and half foreground Dice, pooled by image or batch."""
    if variant not in VARIANTS:
        raise ValueError("Unknown loss variant")
    if logits.ndim != 4 or logits.shape[1] != 1 or target.shape != logits.shape or valid.shape != logits.shape:
        raise ValueError("Expected matching B,1,H,W logits, targets and valid masks")
    target, valid = target.to(logits), valid.to(logits)
    axes = (1, 2, 3)
    counts = valid.sum(axes)
    active = (counts > 0).to(logits.dtype)
    weights = valid
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weights).sum(axes)
    bce = bce / weights.sum(axes).clamp_min(1)
    bce = (bce * active).sum() / active.sum().clamp_min(1)
    probability, truth = logits.sigmoid() * valid, target * valid
    pred_area, gt_area = probability.sum(axes), truth.sum(axes)
    if variant == "per_image":
        dice = 1 - (2 * (probability * truth).sum(axes) + 1e-5) / (pred_area + gt_area + 1e-5)
        dice = (dice * active).sum() / active.sum().clamp_min(1)
    else:
        dice = 1 - (2 * (probability * truth).sum() + 1e-5) / (pred_area.sum() + gt_area.sum() + 1e-5)
    return 0.5 * (bce + dice), {"bce": bce, "dice": dice}
