# Preprocessing (`preprocess.py`)

While looking into random forest options I stumbled upon a problem in our slicing that also plagues the main CNN.
(my fault largely, kind of overlooked this)

slice_segthor.py scaled every CT volume by its own min and max before saving 8-bit PNGs. The max is set by a
handful of extreme voxels (<0.1%, likely metal/artefacts) and ranges from 1892 to 31743 HU across patients.
Result: the soft-tissue range (-100..300 HU), where heart, esophagus and aorta live, got 35 grey levels in
Patient_14 but only 4 in Patient_19. Contrast lost to 8-bit quantisation is gone; no network can recover, with the CNN has to learn to cope with ~10x contrast differences between patients instead of anatomy.

## Window: -160 to 240 HU

Every patient is clipped to the same fixed window and mapped linearly to 0..255.
-160 to 240 is the standard thoracic soft-tissue window (level 40, width 400). Compared side by side with
min-max, lung (-1000..400) and mediastinum (-40..160) windows, it gave the clearest heart/aorta/esophagus
edges, at 1.6 HU per grey level (vs 7.8 for -1000..1000).

Trade-off: everything below -160 HU turns black, including the trachea's air (-650 to -895 HU), so lungs,
trachea and outside air look the same. Try `--hu_min -1000 --hu_max 1000` and compare val Dice.

## Voxel resampling: `--target_spacing 1.0 1.0 2.5`

By default each CT and GT volume is resampled to 1.0 x 1.0 x 2.5 mm before slicing. CT uses linear
interpolation, and GT uses nearest-neighbor interpolation. `stitch.py --preprocess_metadata
data/SEGTHOR/preprocess.json` rebuilds predictions in this processed voxel grid.

## Resizing: `--mm_per_px 0`

By default the full in-plane extent of each resampled slice is resized to 256x256.
`--mm_per_px 1.953125` gives every patient the same physical pixel size (only changes Patients 11, 15, 20),
and `stitch.py --preprocess_metadata data/SEGTHOR/preprocess.json` can undo the crop when rebuilding volumes.

## Run

```
make data/SEGTHOR
```

or `python preprocess.py --dest_dir data/SEGTHOR -p -1`. Val patients: 01, 11, 15, 17, 19.
Rebuild and retrain: earlier results were trained on min-max data.
