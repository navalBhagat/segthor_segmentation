"""Shared native reconstruction for coarse labels and organ crop priors.

The historical filename is retained because existing data and inference modules
reuse these functions. XYZ is native layout; cached scalar planes use ZXY.
"""
from functools import lru_cache
import numpy as np
from scipy.ndimage import map_coordinates, zoom
from skimage.transform import resize


def _shape(values):
    result = tuple(int(v) for v in values)
    if len(result) != 3 or min(result) < 1:
        raise ValueError("Expected three positive shape dimensions")
    return result



def _nearest_resize_indices(source, target):
    # Use the library itself to retain its half-pixel and rounding behavior.
    return resize(np.arange(source, dtype=np.int32), (target,), order=0,
                  preserve_range=True, anti_aliasing=False, mode="constant").astype(np.intp)



def _nearest_zoom_indices(source, target):
    return zoom(np.arange(source, dtype=np.int32), target / source, order=0).astype(np.intp)



def restore_hard_label_volume(labels_zxy, processed_shape_xyz, native_shape_xyz):
    """Hard native XYZ reconstruction without materializing synthetic logits."""
    processed = _shape(processed_shape_xyz)
    native = _shape(native_shape_xyz)
    if labels_zxy.ndim != 3 or labels_zxy.shape[0] != processed[2]:
        raise ValueError("Expected complete processed-Z coverage in Z,X,Y labels")
    low_labels = np.asarray(labels_zxy).transpose(1, 2, 0)
    indices = []
    for axis in range(2):
        to_processed = _nearest_resize_indices(low_labels.shape[axis], processed[axis])
        to_native = _nearest_zoom_indices(processed[axis], native[axis])
        indices.append(to_processed[to_native])
    indices.append(_nearest_zoom_indices(processed[2], native[2]))
    return low_labels[np.ix_(*indices)]



def foreground_bounds(mask, padding=0):
    """Find a box using axis reductions, without a full foreground coordinate list."""
    result = []
    for axis in range(mask.ndim):
        present = np.flatnonzero(mask.any(axis=tuple(i for i in range(mask.ndim) if i != axis)))
        if not len(present):
            return None
        result.append(slice(max(0, int(present[0]) - padding),
                            min(mask.shape[axis], int(present[-1]) + 1 + padding)))
    return tuple(result)



def _zoom_coordinates(source, target):
    # scipy.ndimage.zoom(grid_mode=False) aligns endpoint voxel centers.
    return np.linspace(0.0, source - 1.0, target) if target > 1 else np.zeros(1)



def _xy_restorer(low_shape, processed_shape, native_shape, roi):
    """Prepare exact two-stage XY interpolation, restricted to the requested box."""
    native_coords = [_zoom_coordinates(processed_shape[a], native_shape[a])[roi[a]]
                     for a in range(2)]
    starts = [int(np.floor(c.min())) for c in native_coords]
    stops = [min(processed_shape[a], int(np.ceil(c.max())) + 1)
             for a, c in enumerate(native_coords)]
    low_coords = [(np.arange(starts[a], stops[a], dtype=np.float64) + 0.5)
                  * low_shape[a] / processed_shape[a] - 0.5 for a in range(2)]
    first_grid = np.asarray(np.meshgrid(*low_coords, indexing="ij"))
    second_grid = np.asarray(np.meshgrid(*(native_coords[a] - starts[a] for a in range(2)),
                                        indexing="ij"))

    def restore(plane):
        plane = np.asarray(plane, dtype=np.float32)
        intermediate = map_coordinates(plane, first_grid, order=1,
                                       mode="grid-constant", cval=0, prefilter=False)
        # skimage.resize defaults to clipping to the input range. Its cval
        # exception cannot occur here: full-FOV resize samples at > -0.5 and
        # < size-0.5, so every sample retains some input contribution.
        np.clip(intermediate, plane.min(), plane.max(), out=intermediate)
        return map_coordinates(intermediate, second_grid, order=1,
                               mode="nearest", prefilter=False)

    return restore



def restore_scalar_crop(volume_zxy, processed_shape_xyz, native_shape_xyz,
                        crop_xy, order=1):
    """Restore coarse labels/scalars to a native XY crop and full Z, output ZXY.

    order=0 preserves the input dtype and exact hard reconstruction indices.
    order=1 returns float32, matching full-FOV skimage XY resize followed by
    scipy zoom to native geometry. Crops use native voxel indices, not mm.
    """
    processed, native = _shape(processed_shape_xyz), _shape(native_shape_xyz)
    if volume_zxy.ndim != 3 or volume_zxy.shape[0] != processed[2]:
        raise ValueError("Expected complete processed-Z coverage in Z,X,Y array")
    if order not in (0, 1) or len(crop_xy) != 2:
        raise ValueError("Use order 0 or 1 and two native XY slices")
    roi = []
    for axis, item in enumerate(crop_xy):
        if not isinstance(item, slice) or item.step not in (None, 1):
            raise ValueError("Crop bounds must be contiguous slices")
        start, stop, _ = item.indices(native[axis])
        if start >= stop:
            raise ValueError("Crop must be nonempty")
        roi.append(slice(start, stop))
    roi = (*roi, slice(0, native[2]))
    if order == 0:
        xy = []
        for axis in range(2):
            to_processed = _nearest_resize_indices(volume_zxy.shape[axis + 1], processed[axis])
            to_native = _nearest_zoom_indices(processed[axis], native[axis])[roi[axis]]
            xy.append(to_processed[to_native])
        zs = _nearest_zoom_indices(processed[2], native[2])
        return volume_zxy[np.ix_(zs, *xy)]
    restore_xy = _xy_restorer(volume_zxy.shape[1:], processed, native, roi)

    @lru_cache(maxsize=2)
    def plane(z):
        return restore_xy(volume_zxy[z])

    result = np.empty((native[2], roi[0].stop - roi[0].start,
                       roi[1].stop - roi[1].start), dtype=np.float32)
    for z, coordinate in enumerate(_zoom_coordinates(processed[2], native[2])):
        lo = int(np.floor(coordinate))
        weight = np.float32(coordinate - lo)
        lower = plane(lo)
        result[z] = lower if weight == 0 else lower * (1 - weight) + plane(
            min(lo + 1, processed[2] - 1)) * weight
    return result
