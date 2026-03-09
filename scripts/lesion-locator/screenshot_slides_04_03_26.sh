#!/bin/bash
# Segment/box screenshots for pre/post patients for slides-04-03-26

SCRIPT="scripts/lesion-locator/view_predictions.py"
BASE_DIR="outputs/lesion-locator/screenshots/slides-04-03-26"

PATIENTS=(409 701 488 645 709 683 333)

for id in "${PATIENTS[@]}"; do
  case=$(printf "case_%04d" "$id")
  out_dir="$BASE_DIR/$case"
  for tp in t0 t1; do
    uv run python $SCRIPT $case $tp --mode segment --config box --screenshot --screenshot-dir "$out_dir"
    uv run python $SCRIPT $case $tp --mode segment --config box --screenshot --screenshot-dir "$out_dir" --no-labels
  done
done

echo "Done! Screenshots saved to $BASE_DIR"
