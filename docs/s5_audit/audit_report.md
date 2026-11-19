# VisionPort dataset audit

Annotation file : data\idd\annotations\instances.json
Image folder    : data\idd\images

## 1. Overview
Images: 4 | Annotations: 13 | Categories: 5
Images with no annotation: 0 []
Image sizes in the JSON: 1280x720 (4)

## 2. Class distribution
 id class          count  share
  1 car                3  23.1% ############
  2 motorcycle         3  23.1% ############
  3 pedestrian         2  15.4% ########
  4 bus                3  23.1% ############
  5 truck              2  15.4% ########
Imbalance ratio (largest / smallest class): 1.5 : 1
Rare classes (under 5% of annotations): none

## 3. Annotation validation
No annotation problems found.

## 4. Image files and quality
- file missing: 0 e.g. image ids []
- file cannot be opened (corrupt): 0 e.g. image ids []
- size differs from the JSON: 0 e.g. image ids []
- too dark (mean brightness under 50): 0 e.g. image ids []
- blurry (sharpness under 100): 4 e.g. image ids [1, 2, 3, 4]

## 5. Object size mix (COCO definition by pixel area)
small (under 32x32): 0 | medium (under 96x96): 1 | large: 12

## 6. Summary
Annotations with at least one problem: 0 of 13
Images with at least one problem      : 4 of 4
Suggested actions: remove or repair the flagged annotations, replace missing or corrupt images, review dark and blurry images by eye, and collect more examples of rare classes before training.
