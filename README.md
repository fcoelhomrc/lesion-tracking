# TODO

## `dataset.py`

1. Implement `LongitudinalDataset`
2. Implement a patch-based iterator (see `monai` pkg) for model to ingest the data 
3. Implement `LesionTrackingDataset` (requires registration, connected component analysis, assignment, and possibly graph representation)
4. Implement preprocessing routines (resampling, reorienting, normalization, and data augmentations)
5. Implement caching mechanism
