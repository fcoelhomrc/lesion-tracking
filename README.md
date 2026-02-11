# TODO

## 1. `model.py` — Lightning Model with Flexible 3D Input Handling
- [ ] 3D input → process as sequence of fixed-size 2D masks (slice-wise)
- [ ] 3D input → process as sequence of fixed-size voxels (patch-wise)

## 2. `model.py` — Temporal Aggregation Strategies
- [ ] Single time-point prediction (no temporal context)
- [ ] Two-timepoint aggregation with time embeddings
- [ ] Two-timepoint aggregation with cross-attention

## 3. `trainer.py` — Training Infrastructure
- [ ] PyTorch Lightning trainer
- [ ] WandB integration
- [ ] Hydra-based configs

## 4. `model.py` — V-JEPA-Inspired Approach
- [ ] Self-supervised pre-training (V-JEPA style)
