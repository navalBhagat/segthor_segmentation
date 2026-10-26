#!/bin/bash
set -euo pipefail

DEST=${1:?Usage: postprocess_metrics_plots.sh <result-dir> [gt-root]}
GT_ROOT=${2:-${GT_ROOT:-data/segthor_train/train}}

RAW_VOLUMES="$DEST/volumes/best_epoch/val"
POST_VOLUMES="$DEST/volumes/best_epoch/val_postprocessed"
RAW_METRICS="$DEST/metrics_3d_raw"
POST_METRICS="$DEST/metrics_3d_postprocessed"
PLOTS="$DEST/plots"

apply_aorta_cleanup=0
if [[ -z "${POSTPROCESS_ARGS:-}" ]]; then
  POSTPROCESS_ARGS=(--largest --largest-classes 2)
  apply_aorta_cleanup=1
else
  read -r -a POSTPROCESS_ARGS <<< "$POSTPROCESS_ARGS"
fi

if [[ ! -d "$RAW_VOLUMES" ]]; then
  echo "Missing raw best-epoch volumes: $RAW_VOLUMES" >&2
  exit 1
fi

mkdir -p "$RAW_METRICS" "$POST_METRICS" "$PLOTS" "$DEST/mplconfig"
export MPLCONFIGDIR="$DEST/mplconfig"

echo ">>> Computing raw best-epoch 3D metrics"
python metrics_3d.py \
  --volumes_root "$RAW_VOLUMES" \
  --gt_root "$GT_ROOT" \
  --dest "$RAW_METRICS"

echo ">>> Plotting raw best-epoch 3D metrics"
python plot.py \
  --metric_file "$RAW_METRICS/dice.npz" "$RAW_METRICS/hd95.npz" "$RAW_METRICS/rvd.npz" \
  --dest "$PLOTS/raw_best_epoch_3d_metrics.png" \
  --title "$(basename "$DEST"): raw best-epoch 3D metrics" \
  --headless

echo ">>> Postprocessing best-epoch volumes: ${POSTPROCESS_ARGS[*]}"
python postprocess.py \
  --src "$RAW_VOLUMES" \
  --dest "$POST_VOLUMES" \
  "${POSTPROCESS_ARGS[@]}"

if [[ "$apply_aorta_cleanup" == 1 ]]; then
  echo ">>> Removing aorta components below 0.25 mL (always retaining the largest)"
  # Clean the derived volumes in place, avoiding another full-volume copy.
  python postprocess.py \
    --src "$POST_VOLUMES" \
    --dest "$POST_VOLUMES" \
    --min_ml 0.25 \
    --min-classes 4
fi

echo ">>> Computing postprocessed best-epoch 3D metrics"
python metrics_3d.py \
  --volumes_root "$POST_VOLUMES" \
  --gt_root "$GT_ROOT" \
  --dest "$POST_METRICS"

echo ">>> Plotting postprocessed best-epoch 3D metrics"
python plot.py \
  --metric_file "$POST_METRICS/dice.npz" "$POST_METRICS/hd95.npz" "$POST_METRICS/rvd.npz" \
  --dest "$PLOTS/postprocessed_best_epoch_3d_metrics.png" \
  --title "$(basename "$DEST"): postprocessed best-epoch 3D metrics" \
  --headless
