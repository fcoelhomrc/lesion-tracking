#!/bin/bash
# Screenshot all interesting cases from lesion-locator analysis (bone windowing is default in screenshot mode)

SCRIPT="scripts/lesion-locator/view_predictions.py"
SCREENSHOT_DIR="outputs/lesion-locator/screenshots"

# segment | box
for pair in case_0409:t1 case_0701:t1 case_0488:t1 case_0645:t1 case_0409:t0 case_0709:t1 case_0333:t1 case_0683:t0 case_0715:t1 case_0467:t1 case_0467:t0 case_0294:t0 case_0759:t1 case_0402:t1 case_0523:t1 case_0629:t1 case_0294:t1 case_0709:t0 case_0470:t1 case_0466:t1 case_0428:t1 case_0359:t0 case_0644:t1 case_0516:t0 case_0551:t0 case_0489:t1 case_0341:t1 case_0318:t0 case_0627:t1 case_0191:t0; do
  IFS=: read -r case tp <<< "$pair"
  uv run python $SCRIPT $case $tp --mode segment --config box --screenshot --screenshot-dir $SCREENSHOT_DIR
done

# segment | point
for pair in case_0488:t1 case_0409:t1 case_0409:t0 case_0317:t0 case_0701:t1 case_0317:t1 case_0709:t1 case_0488:t0 case_0249:t1 case_0588:t1 case_0167:t1 case_0788:t0 case_0610:t1 case_0495:t0 case_0194:t1 case_0792:t1 case_0359:t1 case_0527:t1 case_0338:t1 case_0686:t0 case_0441:t0 case_0358:t0 case_0191:t0 case_0359:t0 case_0466:t0 case_0428:t1 case_0489:t1 case_0706:t0 case_0314:t0 case_0191:t1; do
  IFS=: read -r case tp <<< "$pair"
  uv run python $SCRIPT $case $tp --mode segment --config point --screenshot --screenshot-dir $SCREENSHOT_DIR
done

# track | box
for pair in case_0409:t1 case_0409:t0 case_0683:t0 case_0249:t0 case_0701:t1 case_0611:t0 case_0645:t0 case_0473:t0 case_0317:t0 case_0488:t0 case_0489:t0 case_0075:t1 case_0338:t0 case_0345:t0 case_0792:t0 case_0485:t0 case_0328:t0 case_0295:t0 case_0072:t1 case_0628:t0 case_0217:t1 case_0338:t1 case_0359:t1 case_0402:t1 case_0410:t1 case_0414:t1 case_0483:t1 case_0485:t1 case_0516:t1 case_0551:t1; do
  IFS=: read -r case tp <<< "$pair"
  uv run python $SCRIPT $case $tp --mode track --config box --screenshot --screenshot-dir $SCREENSHOT_DIR
done

# track | point
for pair in case_0409:t1 case_0409:t0 case_0317:t0 case_0488:t0 case_0701:t1 case_0683:t0 case_0333:t1 case_0683:t1 case_0317:t1 case_0715:t0 case_0278:t0 case_0670:t1 case_0730:t1 case_0713:t0 case_0362:t0 case_0038:t0 case_0709:t1 case_0259:t0 case_0531:t1 case_0194:t1 case_0191:t1 case_0338:t1 case_0358:t1 case_0359:t1 case_0402:t1 case_0410:t1 case_0414:t1 case_0466:t1 case_0483:t1 case_0485:t1; do
  IFS=: read -r case tp <<< "$pair"
  uv run python $SCRIPT $case $tp --mode track --config point --screenshot --screenshot-dir $SCREENSHOT_DIR
done

echo "Done! Screenshots saved to $SCREENSHOT_DIR"
