# Random forest baseline (rf_baseline.py)

The main idea behind this was to explore an alternative basic ML algorithm, which could be used as an ensemble or a variable for our main NN.

```
python experiments/random_forest/rf_baseline.py
```

## Data

Reads the NIfTI volumes directly: real voxel spacing, and full-resolution GT for the metrics without
going through stitch.py. Uses its own [-1000, 1000] HU window, wider than preprocess.py's -160..240, so the
trachea's air stays distinct, averaged down to 256x256. This RF was written before the general preprocessing fixed HU window fix, so the script does it itself from scratch.

There are 15 features:
- Appearance (11): raw HU, blurred HU at a few scales, edge strength, blob detectors, local texture.
- Position (4): offset from the body centre (scaled by body size, so big and small patients line up),
  height in the scan, depth below the skin.

## Sampling

Up to 10k voxels per class per patient, so the tiny trachea gets as many votes as everything else.
Correcting predictions back to the true class frequencies made the forest too cautious: false
positives dropped (lower HD95), but it labelled 55-80% of every organ as background and Dice fell.
Rejected.

## Evaluation

Same 5 val patients as the CNN split (01, 11, 15, 17, 19). 3D Dice and HD95 on the original 512x512
grid with real voxel spacing. Classes 1-3 only: the aorta does not exist in this data release
(it is merged into "esophagus").

## Verdict (val, mean +- std)

|           | esophagus+aorta | heart        | trachea      |
|-----------|-----------------|--------------|--------------|
| Dice      | 0.48 +- 0.10    | 0.66 +- 0.18 | 0.59 +- 0.09 |
| HD95 (mm) | 48 +- 26        | 66 +- 31     | 65 +- 23     |

Respectable for a model that never sees more than one pixel at a time, not competitive with a CNN.
The interesting bit: position beats appearance, 59% to 41% of feature importance, with depth below
the skin as the top feature. Knowing where you are matters more than what the pixel looks like,
which is an argument for giving the CNN more spatial context. In the end the results are a bit too weak, ensembling would harm the performance
of our main model. But it definitely gives us a better understanding of what matters in prediction.

## Don't trust the confusion matrix at first glance

The row-normalised confusion matrix looks great (0.87-0.96 on the diagonal), but rows only show recall:
how much of each organ was found. Precision tells the other half:

|                 | recall | precision | predicted vs true volume |
|-----------------|--------|-----------|--------------------------|
| esophagus+aorta | 0.87   | 0.35      | 2.5x                     |
| heart           | 0.91   | 0.51      | 1.8x                     |
| trachea         | 0.96   | 0.45      | 2.1x                     |

The forest finds almost every organ voxel, but also labels about as much again that isn't organ.
91-98% of those false positives come from background. Background is 59M voxels, so its "0.99" still
hides ~630k wrong voxels, more than all true organ voxels combined. These scattered false positives
around the mediastinum are also why HD95 is so high: the forest has no idea an organ is one connected shape.

One cheap idea worth testing after the CNN is retrained: use the forest as a veto. With 0.87-0.96 recall,
a CNN organ prediction where the forest says background is probably wrong, so zero it out.

**Takeaway:** the forest finds ~90% of every organ but doubles its volume with false positives, so it's a baseline and maybe a CNN veto, not an ensemble partner.
