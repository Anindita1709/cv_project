# AirRoom PyTorch Reimplementation

This is a clean, repo-structured PyTorch reimplementation of the AirRoom pipeline.

## Structure

- `preprocess.py` builds the reference database
- `inference.py` runs coarse-to-fine room re-identification
- `models/build_model.py` contains model builders
- `data/query_dataset.py` and `data/reference_dataset.py` scan the dataset
- `utils/geometry.py` implements Delaunay-based receptive field expansion
- `utils/scoring.py` implements mutual nearest neighbor scoring

## Dataset layout

```text
.datasets/
  ReplicaReID/
    scene_001/
      room_001/
        rgb/
        depth/
      room_002/
        rgb/
        depth/
    room_label.txt
```

After preprocessing, each room gets:

```text
ref/
  rgb/0.png
  embed/room_feature.pt
  embed/objects.pt
  embed/patches.pt
```

## Install

```bash
pip install -r requirements.txt
```

## Preprocess

```bash
python preprocess.py
```

## Inference

```bash
python inference.py
```

## Notes

- Default segmentation is `maskrcnn` for easy reproducibility.
- LightGlue is optional. If unavailable, inference falls back to ORB matching.
- The original AirRoom repo uses CLIP for reference-image selection, DINOv2/AnyLoc-style global retrieval, object-aware scoring, and LightGlue-style fine matching. This reimplementation preserves that pipeline structure while keeping the code easier to run and modify.
